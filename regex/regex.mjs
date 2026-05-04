const REGEX_MEMORY_ERROR = -6;
const LABEL_NO_MATCH_ERROR = -5;
const SET_TOKEN_BODY = 1;
const SET_TOKEN_LAST = 2;
const DEFAULT_GROUP_MOD = " ".charCodeAt(0);
const encoder = new TextEncoder();
const decoder = new TextDecoder();

export class RegexError extends Error {}

export async function createRegex(options = {}) {
  const { default: createWasm } = await import("./build/regex_wasm.js?v=23");
  return new Regex(await createWasm(options));
}

export default createRegex;

export class Regex {
  constructor(module) {
    this.module = module;
  }

  translate(regex, { caseSensitive = true } = {}) {
    regex = String(regex);
    return caseSensitive ? regex : addCasePairs(regex);
  }

  match(regex, string, options = {}) {
    const pattern = this.translate(regex, options);
    return this.#withStrings(pattern, string, (re, text) => {
      const start = this.module._malloc(4);
      const end = this.module._malloc(4);
      try {
        this.module._match(re, text, start, end);
        return this.#returnValue(pattern, this.#i32(start), this.#i32(end));
      } finally {
        this.module._free(start);
        this.module._free(end);
      }
    });
  }

  matcha(regex, string, options = {}) {
    const pattern = this.translate(regex, options);
    return this.#withStrings(pattern, string, (re, text) => {
      const nPtr = this.module._malloc(4);
      const startsPtr = this.module._malloc(4);
      const endsPtr = this.module._malloc(4);
      let n, starts, ends;
      try {
        this.module._matcha(re, text, nPtr, startsPtr, endsPtr);
        n = this.#i32(nPtr);
        starts = this.#i32(startsPtr);
        ends = this.#i32(endsPtr);
      } finally {
        this.module._free(nPtr);
        this.module._free(startsPtr);
        this.module._free(endsPtr);
      }
      if (n == 0) return [[], []];
      if (n == REGEX_MEMORY_ERROR) throw new Error("Failed memory allocation.");
      if (n == -2) throw new TypeError("matcha requires a nonempty string.");
      if (n == -1) {
        try {
          const result = this.#returnValue(pattern, this.#i32(starts), this.#i32(ends));
          if (result == null) return [[], []];
          if (result[0] == 0 && result[1] == 0) throw new RegexError("matcha requires nonempty regex.");
          throw new RegexError("Unexpected matcha error.");
        } finally {
          if (starts) this.module._free(starts);
        }
      }
      const out = [
        Array.from(this.module.HEAP32.subarray(starts >> 2, (starts >> 2) + n)),
        Array.from(this.module.HEAP32.subarray(ends >> 2, (ends >> 2) + n)),
      ];
      this.module._free(starts);
      return out;
    });
  }

  label(regex, string, options = {}) {
    const pattern = this.translate(regex, options);
    return this.#withStrings(pattern, string, (re, text) => {
      const labelsPtr = this.module._malloc(4);
      const groupsPtr = this.module._malloc(4);
      const spansPtr = this.module._malloc(4);
      const n = this.module._label(re, text, labelsPtr, groupsPtr, spansPtr);
      const labels = this.#i32(labelsPtr);
      const groups = this.#i32(groupsPtr);
      const spans = this.#i32(spansPtr);
      this.module._free(labelsPtr);
      this.module._free(groupsPtr);
      this.module._free(spansPtr);
      if (n == LABEL_NO_MATCH_ERROR) return null;
      if (n == REGEX_MEMORY_ERROR) throw new Error("Failed memory allocation.");
      if (n < 0) throw new RegexError(`Invalid regular expression (code ${-n}).`);
      const compiled = this.compile(regex, options);
      const result = {
        labels: Array.from(this.module.HEAP32.subarray(labels >> 2, (labels >> 2) + n)),
        groups: Array.from(this.module.HEAP32.subarray(groups >> 2, (groups >> 2) + n)),
        spans: compiled.groupSpans,
      };
      if (labels) this.module._free(labels);
      if (groups) this.module._free(groups);
      if (spans) this.module._free(spans);
      return result;
    });
  }

  compile(regex, options = {}) {
    const pattern = this.translate(regex, options);
    return this.#withString(pattern, re => {
      const tokensPtr = this.module._malloc(4);
      const groupsPtr = this.module._malloc(4);
      this.module._regex_count(re, tokensPtr, groupsPtr);
      const nTokens = this.#i32(tokensPtr);
      const nGroups = this.#i32(groupsPtr);
      this.module._free(tokensPtr);
      this.module._free(groupsPtr);
      if (nGroups == REGEX_MEMORY_ERROR) throw new Error("Failed memory allocation.");
      if (nTokens < 0) this.#returnValue(pattern, nTokens, nGroups);
      if (nTokens == 0) return { regex: pattern, tokens: [], jumps: [], jumpf: [], jumpi: [], groups: [], groupSpans: [] };

      const jumps = this.module._malloc(4 * nTokens);
      const jumpf = this.module._malloc(4 * nTokens);
      const tokens = this.module._malloc(nTokens + 1);
      const jumpi = this.module._malloc(nTokens + 1);
      try {
        this.module._regex_set_jump(re, nTokens, nGroups, tokens, jumps, jumpf, jumpi);
        if (this.#i32(jumps) == REGEX_MEMORY_ERROR) throw new Error("Failed memory allocation.");
        const tokenBytes = Array.from(this.module.HEAPU8.subarray(tokens, tokens + nTokens));
        const jumpFlags = Array.from(this.module.HEAPU8.subarray(jumpi, jumpi + nTokens));
        return {
          regex: pattern,
          tokens: tokenBytes.map((byte, i) => ({
            index: i,
            byte,
            text: safeByte(byte),
            set: jumpFlags[i] == SET_TOKEN_BODY || jumpFlags[i] == SET_TOKEN_LAST,
          })),
          jumps: Array.from(this.module.HEAP32.subarray(jumps >> 2, (jumps >> 2) + nTokens)),
          jumpf: Array.from(this.module.HEAP32.subarray(jumpf >> 2, (jumpf >> 2) + nTokens)),
          jumpi: jumpFlags,
          groups: tokenGroups(pattern, nTokens, nGroups),
          groupSpans: groupSpans(pattern, nGroups),
        };
      } finally {
        this.module._free(jumps);
        this.module._free(jumpf);
        this.module._free(tokens);
        this.module._free(jumpi);
      }
    });
  }

  #returnValue(regex, start, end) {
    if (start >= 0) return [start, end];
    if (end == 0) return null;
    if (end == -1) return [0, 0];
    if (end == REGEX_MEMORY_ERROR) throw new Error("Failed memory allocation.");
    if (end < -1) throw new RegexError(`Invalid regular expression (code ${-end})${start < -1 ? ` at position ${-start - 1}` : ""}.`);
    return null;
  }

  #withStrings(regex, string, fn) {
    return this.#withString(regex, re => this.#withString(string, text => fn(re, text)));
  }

  #withString(value, fn) {
    const bytes = encoder.encode(String(value));
    const ptr = this.module._malloc(bytes.length + 1);
    this.module.HEAPU8.set(bytes, ptr);
    this.module.HEAPU8[ptr + bytes.length] = 0;
    try { return fn(ptr); }
    finally { this.module._free(ptr); }
  }

  #i32(ptr) {
    return this.module.HEAP32[ptr >> 2];
  }
}

export function safeByte(byte) {
  if (byte == 10) return "\\n";
  if (byte == 9) return "\\t";
  if (byte == 13) return "\\r";
  if (byte == 0) return "\\0";
  if (byte < 128) return decoder.decode(Uint8Array.of(byte));
  return "\\x" + byte.toString(16).padStart(2, "0");
}

function addCasePairs(regex) {
  let out = "";
  let inSet = false;
  for (let i = 0; i < regex.length; i++) {
    const c = regex[i];
    if (c == "[" && !inSet) inSet = true;
    else if (c == "]" && inSet) inSet = false;
    if (!/[a-z]/i.test(c)) out += c;
    else if (inSet) out += c + pairCase(c);
    else out += `[${c}${pairCase(c)}]`;
  }
  return out;
}

function pairCase(c) {
  return c == c.toLowerCase() ? c.toUpperCase() : c.toLowerCase();
}

function groupSpans(regex, nGroups) {
  const spans = Array(2 * nGroups).fill(-1);
  const giStack = [];
  const sStack = [];
  let gi = -1, cgs = "", ng = 0;
  for (let i = 0; i < regex.length; i++) {
    const token = regex[i];
    if ("([{".includes(token) && cgs != "[") {
      gi = ng++;
      cgs = token;
      giStack.push(gi);
      sStack.push(token);
      spans[2 * gi] = spans[2 * gi + 1] = gi;
    } else if (giStack.length && closes(cgs, token)) {
      spans[2 * gi + 1] = ng - 1;
      giStack.pop();
      sStack.pop();
      gi = giStack.length ? giStack.at(-1) : -1;
      cgs = sStack.length ? sStack.at(-1) : "";
    }
  }
  return spans;
}

function tokenGroups(regex, nTokens, nGroups) {
  const mods = Array(nGroups).fill(DEFAULT_GROUP_MOD);
  const spans = groupSpans(regex, nGroups);
  const giStack = [];
  const sStack = [];
  let gi = -1, cgs = "", ng = 0;
  for (let i = 0; i < regex.length; i++) {
    const token = regex[i];
    if ("([{".includes(token) && cgs != "[") {
      gi = ng++;
      cgs = token;
      giStack.push(gi);
      sStack.push(token);
    } else if (giStack.length && closes(cgs, token)) {
      if ("*?|".includes(regex[i + 1] || "")) mods[gi] = regex.charCodeAt(i + 1);
      giStack.pop();
      sStack.pop();
      gi = giStack.length ? giStack.at(-1) : -1;
      cgs = sStack.length ? sStack.at(-1) : "";
    }
  }
  const groups = [];
  giStack.length = sStack.length = 0;
  gi = -1; cgs = ""; ng = 0;
  for (let i = 0; i < regex.length && groups.length < nTokens; i++) {
    const token = regex[i];
    if ("([{".includes(token) && cgs != "[") {
      gi = ng++;
      cgs = token;
      giStack.push(gi);
      sStack.push(token);
      if (mods[gi] != DEFAULT_GROUP_MOD) groups.push(gi);
    } else if (giStack.length && closes(cgs, token)) {
      giStack.pop();
      sStack.pop();
      gi = giStack.length ? giStack.at(-1) : -1;
      cgs = sStack.length ? sStack.at(-1) : "";
    } else {
      const next = regex[i + 1] || "";
      if (cgs != "[" && "*?|".includes(next)) {
        groups.push(gi);
        i++;
      }
      if (cgs == "[" || !"*?|".includes(token)) groups.push(gi);
    }
  }
  return groups.map((group, token) => ({ token, group, span: group < 0 ? [-1, -1] : spans.slice(2 * group, 2 * group + 2) }));
}

function closes(start, end) {
  return (start == "(" && end == ")") || (start == "[" && end == "]") || (start == "{" && end == "}");
}
