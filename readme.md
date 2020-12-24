
%% {%- include profile.html -%}

## About Me

I specialize in artificial intelligence / machine learning research and engineering. I write high performance code and develop mathematical theory. My unique skills are in distribution outcome prediction, interpretable models, black box optimization, and models that work well with small amounts of data.


<p style="margin-bottom:40px;"></p>
<hr>

# Publications

Alternate listings at [Google Scholar](https://scholar.google.com/citations?user=wamfO3sAAAAJ&hl=en), [DBLP](https://dblp.org/pers/hd/l/Lux:Thomas) (imperfect matches), and [ORCID](https://orcid.org/0000-0002-1858-4724) 0000-0002-1858-4724.

{% assign pending = 0 %}
{% for item in site.publications %} {% if item.status != 'published' and item.status != 'unpublished' and item.id != 'example' %}
    {% assign pending = pending | plus: 1 %}
{% endif %} {% endfor %}

{% if pending > 0 %}
<p style="margin-bottom:30px;"></p>
## Under Review

{% for item in site.publications %} {% if item.status != 'published' and item.status != 'accepted' and item.id != 'example' %}

#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
<span style="font-size: 10pt;">
{{item.authors}}. {{item.venue}}. {% if item.status != nil %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.* {% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.code != nil %} [[code]({{item.code}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} {% if item.video != nil %} [[video]({{item.video}})] {% endif %}
</span>
{% endif %} {% endfor %}
{% endif %}

<p style="margin-bottom:30px;"></p>
## Published

{% for item in site.publications %} {% if item.status == 'published' or item.status == 'accepted' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %} 
<span style="font-size: 10pt;">
{{item.authors}}. {{item.venue}}. {% if item.status != 'published' %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.* {% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.code != nil %} [[code]({{item.code}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} {% if item.video != nil %} [[video]({{item.video}})] {% endif %}
</span>
{% endif %} {% endfor %}

<p style="margin-bottom:40px;"></p>
<hr>

# Projects

{% for item in site.projects %}
#### {{item.title}}
<span style="font-size: 10pt;">
{{item.authors}}. {{item.venue}}. *{{item.month}}, {{item.year}}.* {% if item.pdf != nil %} [[paper]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.code != nil %} [[code]({{item.code}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} {% if item.video != nil %} [[video]({{item.video}})] {% endif %}
</span>
{% endfor %}

<p style="margin-bottom:40px;"></p>
<hr>

# Education

## Virginia Tech

I received my Ph.D. in Computer Science (CS) from [Virginia Tech](https://vt.edu) in August 2020. I was co-advised by [Dr. Layne T. Watson](https://dac.cs.vt.edu/person/layne-t-watson-2/) in CS and [Dr. Yili Hong](https://www.apps.stat.vt.edu/hong) in Statistics. I worked as part of the [VarSys research team](http://varsys.cs.vt.edu), applying mathematical models to the study of variability in computation. My primary research area was computational science, specifically numerical analysis and approximation theory. My dissertation is titled [Interpolants, Error Bounds, and Mathematical Software for Modeling and Predicting Variability in Computer Systems](https://vtechworks.lib.vt.edu/handle/10919/100059).

## Roanoke College

I received my B.S. in Computer Science with minors in Mathematics and Physics from [Roanoke College](https://roanoke.edu). I was advised by [Dr. Durell Bouchard](https://directory.roanoke.edu/faculty?username=bouchard) and did research projects with him in robotics and computer vision; I did research with [Dr. Anil Shende](https://directory.roanoke.edu/faculty?username=shende) on parallel computing and machine learning projects. At Roanoke I was heavily involved in student life and cocurriculars, committing three years (and lots of time) to Student Government and being a Resident Advisor, as well as participating in every club and organization whose meetings I could attend.


# Personal

My personal research ambitions outside of my direct experience and dissertation work focus on progress towards artificial general intelligence. My current line of work for this is centered about trying to automatically infer the structure of data in order to build very sample-efficient approximations.

Outside of Computer Science, I like to hike, run, ride motorcycles, and play music. I was originally trained in percussion / drum set for Jazz and I've taught myself some guitar and piano for fun on the side. I ventured into positive psychology in college and wrote [this summary of Jonathan Haidt's *Happiness Hypothesis*](https://tchlux.github.io/documents/Happiness.txt.html), I love that book! Lastly, I enjoy trying to take pretty pictures in my free time and [these are some of my favorites](https://www.icloud.com/sharedalbum/#B0JGWZuqDpCayQ).

<div align="center"><img src="https://tchlux.github.io/media/blacksburg-sunset-with-bird.jpeg" width="90%" style="border-radius:10px; display:block;"></div>
