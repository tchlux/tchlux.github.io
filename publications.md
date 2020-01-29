---
title: Publications
---

[Google Scholar](https://scholar.google.com/citations?user=wamfO3sAAAAJ&hl=en), [DBLP](https://dblp.org/pers/hd/l/Lux:Thomas) (imperfect matches), and [ORCID](https://orcid.org/0000-0002-1858-4724): 0000-0002-1858-4724

<p style="margin-bottom:50px;"></p>

# Published

{% assign publications = site.publications | where:'status','published' %}
{% for item in publications %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. *{% if item.status != 'published' %} ({{item.status}}) {% endif %} {{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %} 
{% endfor %}

{% assign pending = '' | split '' %}
{% for item in site.publications %} {% if item.status != 'published' %}
    {% assign pending = pending | push: item %}
{% endif %} {% endfor %}
{% if pending.size > 0 %}

<p style="margin-bottom:100px;"></p>

# Pending

{% for item in pending %}
#### {{item.title}} {% if item.type != 'paper' %} ({{item.type}}) {% endif %}
{{item.authors}}.
<br>{{item.venue}}. *{% if item.status != nil %} **{{item.status}}** {% endif %} {{item.month}}, {{item.year}}.*
{% if item.pdf != nil %} [[pdf]({{item.pdf}})] {% endif %} {% if item.link != nil %} [[link]({{item.link}})] {% endif %} {% if item.slides != nil %} [[slides]({{item.slides}})] {% endif %}
{% endfor %}

{% endif %}
