---
title: Publications
---

- ORCID: 0000-0002-1858-4724
- [Google Scholar](https://scholar.google.com/citations?user=wamfO3sAAAAJ&hl=en)
- [DBLP](https://dblp.org/pers/hd/l/Lux:Thomas) (imperfect matches)

<p style="margin-bottom:50px;"></p>

# First Author Publications

{% assign publications=site.publications | where:"authorship","first" %}
{% for item in publications %}
#### {{item.title}}
{{item.authors}}
{{item.venue}}. *{{item.month}}, {{item.year}}*
{% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %}
{% if item.link != nil %} [[link]({{item.link}})] {% endif %}
{% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %}
{% endfor %}

<p style="margin-bottom:100px;"></p>

# Co-authored Publications

{% assign publications=site.publications | where:"authorship","coauthor" %}
{% for item in publications %}
#### {{item.title}}
{{item.authors}}
{{item.venue}}. *{{item.month}}, {{item.year}}*
{% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %}
{% if item.link != nil %} [[link]({{item.link}})] {% endif %}
{% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %}
{% endfor %}

