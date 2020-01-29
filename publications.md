---
title: Publications
---

Alternate listings at [Google Scholar](https://scholar.google.com/citations?user=wamfO3sAAAAJ&hl=en), [DBLP](https://dblp.org/pers/hd/l/Lux:Thomas) (imperfect matches), and [ORCID](https://orcid.org/0000-0002-1858-4724) 0000-0002-1858-4724.

<p style="margin-bottom:50px;"></p>

# Publications

{% for item in site.publications %} {% if item.status == 'published' or item.status == 'accepted' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. {% if item.status != 'published' %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} 
{% endif %} {% endfor %}

{% assign pending = 0 %}
{% for item in site.publications %} {% if item.status != 'published' and item.id != 'example' %}
    {% assign pending = pending | plus: 1 %}
{% endif %} {% endfor %}
{% if pending > 0 %}

<p style="margin-bottom:100px;"></p>

# Pending

{% for item in site.publications %} {% if item.status != 'published' and item.status != 'accepted' and item.id != 'example' %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. {% if item.status != nil %} **{{item.status}}** {% endif %} *{{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %}
{% endif %} {% endfor %}

{% endif %}
