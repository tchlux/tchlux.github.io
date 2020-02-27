
<p align="left" style="margin-left: min(10vw, 50px)">
  <img src="https://avatars1.githubusercontent.com/u/7769932?s=460&v=4" width="200" style="border-radius:10%">
</p>


I specialize in artificial intelligence / machine learning research and engineering. I write high performance code and develop mathematical theory. My unique skills are in distribution outcome prediction, interpretable models, black box optimization, and models that work well with small amounts of data. I am actively looking for jobs in the San Francisco Bay area.

I have an extensive set of co-curricular projects that have resulted in publications, granting me experiences that are not explicitly listed in my job history. Please see my GitHub and publication lists for more details on these works. 

## At Virginia Tech

I am a 4th year Ph.D. candidate in Computer Science (CS) at [Virginia Tech](https://vt.edu) co-advised by [Dr. Layne T. Watson](https://dac.cs.vt.edu/person/layne-t-watson-2/) in CS and [Dr. Yili Hong](https://www.apps.stat.vt.edu/hong) in Statistics. I work as part of the [VarSys research team](http://varsys.cs.vt.edu), applying mathematical models to the study of variability in computation. I plan to graduate in 2020 and my primary research area is computational science, specifically numerical analysis and approximation theory. My dissertation is on writing mathematical software that constructs piecewise quintic Hermite interpolating polynomials.


<p style="margin-bottom:100px;"></p>
<hr>



# Publications

Alternate listings at [Google Scholar](https://scholar.google.com/citations?user=wamfO3sAAAAJ&hl=en), [DBLP](https://dblp.org/pers/hd/l/Lux:Thomas) (imperfect matches), and [ORCID](https://orcid.org/0000-0002-1858-4724) 0000-0002-1858-4724.

{% assign pending = 0 %}
{% for item in site.publications %} {% if item.status != 'published' and item.id != 'example' %}
    {% assign pending = pending | plus: 1 %}
{% endif %} {% endfor %}

{% if pending > 0 %}
<p style="margin-bottom:20px;"></p>
## Pending

{% for item in site.publications %} {% if item.status != 'published' and item.status != 'accepted' and item.id != 'example' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. {% if item.status != nil %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %}
{% endif %} {% endfor %}
{% endif %}

<p style="margin-bottom:20px;"></p>
## Published

{% for item in site.publications %} {% if item.status == 'published' or item.status == 'accepted' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. {% if item.status != 'published' %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} 
{% endif %} {% endfor %}

<p style="margin-bottom:30px;"></p>
<hr>


# Personal Pursuits

My personal research ambitions outside of my direct experience and dissertation work focus on progress towards artificial general intelligence. My current line of work for this is centered about trying to automatically infer the structure of data in order to build very sample-efficient approximations.

Outside of Computer Science, I like to hike, run, ride motorcycles, and play music. I was originally trained in percussion / drum set for Jazz and I've taught myself some guitar and piano for fun on the side. As many others do, I enjoy trying to take pretty pictures in my free time. [These are some of my favorites](https://www.icloud.com/sharedalbum/#B0JGWZuqDpCayQ).

