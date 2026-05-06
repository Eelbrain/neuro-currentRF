{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}


{% block methods %}
{% if methods %}

.. rubric:: Methods

.. autosummary::
   :toctree: generated

{% for item in methods %}
   {%- if not item.startswith('_') or item in ['__call__'] %}
   ~{{ name }}.{{ item }}
   {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}
