
<p align="left" style="margin-left: min(10vw, 50px)">
  <div>
  <img src="https://avatars1.githubusercontent.com/u/7769932?s=460&v=4" width="200" style="border-radius:10%; display:block;">
  {%- assign social = site.minima.social_links -%}
  <ul class="social-media-list" style="dislpay: inline;">
    {%- if social.gscholar -%}<li style="padding: 0px 0px 0px 5px; display: inline-block; width:35px; height:35px;"><a href="https://scholar.google.com/citations?user={{ social.gscholar | cgi_escape | escape }}&hl=en" title="Google Scholar {{ social.gscholar | escape }}"><svg height="30px" width="30px" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><g fill="none" fill-rule="evenodd"><path d="M256 411.12L0 202.667 256 0z" fill="#4285f4"/><path d="M256 411.12l256-208.453L256 0z" fill="#356ac3"/><circle cx="256" cy="362.667" fill="#a0c3ff" r="149.333"/><path d="M121.037 298.667c23.968-50.453 75.392-85.334 134.963-85.334s110.995 34.881 134.963 85.334H121.037z" fill="#76a7fa"/></g></svg></a></li>{%- endif -%}
    {%- if social.stoverflow -%}<li style="padding: 0px 0px 0px 5px; display: inline-block; width:35px; height:35px;"><a href="https://stackoverflow.com/users/{{ social.stoverflow | cgi_escape | escape }}" title="Stack Overflow {{ social.stoverflow | escape }}"><svg height="34px" width="34px" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><style>.st0{fill:#bcbbbb}.st1{fill:#f48023}</style><path class="st0" d="M84.4 93.8V70.6h7.7v30.9H22.6V70.6h7.7v23.2z"/><path class="st1" d="M38.8 68.4l37.8 7.9 1.6-7.6-37.8-7.9-1.6 7.6zm5-18l35 16.3 3.2-7-35-16.4-3.2 7.1zm9.7-17.2l29.7 24.7 4.9-5.9-29.7-24.7-4.9 5.9zm19.2-18.3l-6.2 4.6 23 31 6.2-4.6-23-31zM38 86h38.6v-7.7H38V86z"/></svg></a></li>{%- endif -%}
    {%- if social.github -%}<li style="padding: 0px 0px 0px 5px; display: inline-block; width:35px; height:35px;"><a style="width:35px; height:35px; display:inline-block;" href="https://github.com/{{ social.github | cgi_escape | escape }}" title="Github {{ social.github | escape }}"><svg style="width:30px; height:30px;" viewBox="0 0 20 20" class="svg-icon grey"><use xlink:href="{{ '/assets/minima-social-icons.svg#github' | relative_url }}"></use></svg></a></li>{%- endif -%}
    {%- if social.linkedin -%}<li style="padding: 0px 0px 0px 5px; display: inline-block; width:35px; height:35px;"><a href="https://www.linkedin.com/in/{{ social.linkedin | cgi_escape | escape }}" title="LinkedIn {{ social.linkedin | escape }}"><svg height="30" width="30" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><g fill="none"><path d="M0 18.338C0 8.216 8.474 0 18.92 0h218.16C247.53 0 256 8.216 256 18.338v219.327C256 247.79 247.53 256 237.08 256H18.92C8.475 256 0 247.791 0 237.668V18.335z" fill="#069"/><path d="M77.796 214.238V98.986H39.488v115.252H77.8zM58.65 83.253c13.356 0 21.671-8.85 21.671-19.91-.25-11.312-8.315-19.915-21.417-19.915-13.111 0-21.674 8.603-21.674 19.914 0 11.06 8.312 19.91 21.169 19.91h.248zM99 214.238h38.305v-64.355c0-3.44.25-6.889 1.262-9.346 2.768-6.885 9.071-14.012 19.656-14.012 13.858 0 19.405 10.568 19.405 26.063v61.65h38.304v-66.082c0-35.399-18.896-51.872-44.099-51.872-20.663 0-29.738 11.549-34.78 19.415h.255V98.99H99.002c.5 10.812-.003 115.252-.003 115.252z" fill="#fff"/></g></svg></a></li>{%- endif -%}
    {%- if social.twitter -%}<li style="padding: 0px 0px 0px 5px; display: inline-block; width:35px; height:35px;"><a href="https://twitter.com/{{ social.twitter | cgi_escape | escape }}" title="Twitter {{ social.twitter | escape }}"><svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 300.00006 244.18703" height="30" width="30"> <g transform="translate(-539.17946,-568.85777)" id="layer1"> <path id="path3611" style="fill:#1da1f2;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 633.89823,812.04479 c 112.46038,0 173.95627,-93.16765 173.95627,-173.95625 0,-2.64628 -0.0539,-5.28062 -0.1726,-7.90305 11.93799,-8.63016 22.31446,-19.39999 30.49762,-31.65984 -10.95459,4.86937 -22.74358,8.14741 -35.11071,9.62551 12.62341,-7.56929 22.31446,-19.54304 26.88583,-33.81739 -11.81284,7.00307 -24.89517,12.09297 -38.82383,14.84055 -11.15723,-11.88436 -27.04079,-19.31655 -44.62892,-19.31655 -33.76374,0 -61.14426,27.38052 -61.14426,61.13233 0,4.79784 0.5364,9.46458 1.58538,13.94057 -50.81546,-2.55686 -95.87353,-26.88582 -126.02546,-63.87991 -5.25082,9.03545 -8.27852,19.53111 -8.27852,30.73006 0,21.21186 10.79366,39.93837 27.20766,50.89296 -10.03077,-0.30992 -19.45363,-3.06348 -27.69044,-7.64676 -0.009,0.25652 -0.009,0.50661 -0.009,0.78077 0,29.60957 21.07478,54.3319 49.0513,59.93435 -5.13757,1.40062 -10.54335,2.15158 -16.12196,2.15158 -3.93364,0 -7.76596,-0.38716 -11.49099,-1.1026 7.78383,24.2932 30.35457,41.97073 57.11525,42.46543 -20.92578,16.40207 -47.28712,26.17062 -75.93712,26.17062 -4.92898,0 -9.79834,-0.28036 -14.58427,-0.84634 27.05868,17.34379 59.18936,27.46396 93.72193,27.46396" /> </g> </svg></a></li>{%- endif -%}
  </ul>
  </div>
</p>


I specialize in artificial intelligence / machine learning research and engineering. I write high performance code and develop mathematical theory. My unique skills are in distribution outcome prediction, interpretable models, black box optimization, and models that work well with small amounts of data. I am actively looking for jobs in the San Francisco Bay area.

I have an extensive set of co-curricular projects that have resulted in publications, granting me experiences that are not explicitly listed in my job history. Please see my GitHub and publication lists for more details on these works. 

## At Virginia Tech

I am a 4th year Ph.D. candidate in Computer Science (CS) at [Virginia Tech](https://vt.edu) co-advised by [Dr. Layne T. Watson](https://dac.cs.vt.edu/person/layne-t-watson-2/) in CS and [Dr. Yili Hong](https://www.apps.stat.vt.edu/hong) in Statistics. I work as part of the [VarSys research team](http://varsys.cs.vt.edu), applying mathematical models to the study of variability in computation. I plan to graduate in 2020 and my primary research area is computational science, specifically numerical analysis and approximation theory. My dissertation is on writing mathematical software that constructs piecewise quintic Hermite interpolating polynomials.


<p style="margin-bottom:40px;"></p>
<hr>



# Publications

Alternate listings at [Google Scholar](https://scholar.google.com/citations?user=wamfO3sAAAAJ&hl=en), [DBLP](https://dblp.org/pers/hd/l/Lux:Thomas) (imperfect matches), and [ORCID](https://orcid.org/0000-0002-1858-4724) 0000-0002-1858-4724.

{% assign pending = 0 %}
{% for item in site.publications %} {% if item.status != 'published' and item.id != 'example' %}
    {% assign pending = pending | plus: 1 %}
{% endif %} {% endfor %}

{% if pending > 0 %}
<p style="margin-bottom:30px;"></p>
## Pending

{% for item in site.publications %} {% if item.status != 'published' and item.status != 'accepted' and item.id != 'example' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. {% if item.status != nil %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %}
{% endif %} {% endfor %}
{% endif %}

<p style="margin-bottom:30px;"></p>
## Published

{% for item in site.publications %} {% if item.status == 'published' or item.status == 'accepted' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. {% if item.status != 'published' %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} 
{% endif %} {% endfor %}

<p style="margin-bottom:40px;"></p>
<hr>


# Personal Pursuits

My personal research ambitions outside of my direct experience and dissertation work focus on progress towards artificial general intelligence. My current line of work for this is centered about trying to automatically infer the structure of data in order to build very sample-efficient approximations.

Outside of Computer Science, I like to hike, run, ride motorcycles, and play music. I was originally trained in percussion / drum set for Jazz and I've taught myself some guitar and piano for fun on the side. As many others do, I enjoy trying to take pretty pictures in my free time. [These are some of my favorites](https://www.icloud.com/sharedalbum/#B0JGWZuqDpCayQ).

