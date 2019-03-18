---
layout: post
comments: true
title: Predicción del sentimiento usando Python
---

Actualmente las redes sociales se han convertido en una fuente de datos masivos.
Esto hace que grandes compañías traten de entender a sus usuarios mediante el análisis de dichas redes. 
En este post trataremos de elaborar un modelo que nos permita predecir el sentimiento de un usuario al manifestar su opinión en Twitter.

# Dataset
En primer lugar es necesario el uso de un Dataset como el proporcionado por [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment#database.sqlite).
Este corpus consiste en un conjunto de 14640 tweets etiquetados según el sentimiento del usuario (negativo, positivo, neutro).
Para cada tweet se disponen de los siguientes datos:
* tweet_id
* airline_sentiment
* airline_sentiment_confidence
* negativereason
* negativereason_confidence
* airline
* airline_sentiment_gold
* name
* negativereason_gold
* retweet_count
* text
* tweet_coord
* tweet_created
* tweet_location
* user_timezone

En nuestro caso nos centraremos en los campos airline_sentiment y text.

#Preproceso
Se ha decidido elaborar un preproceso del texto convirtiendo cada tweet a minúsculas ademas de sustituir las urls por la etiqueta <url> y las menciones por la etiqueta <mention>. Para ello se ha utilizado el siguiente script:
{% highlight python %}
url_re = re.compile("https?://[^\s]+")
mention_re = re.compile("@(\w+)")
def preprocessing(text):
    text_clean = url_re.sub("<url>",text)
    text_clean = mention_re.sub("<mention>", text_clean)
    text_clean = text_clean.lower()
    return text_clean 
{% endhighlight %}
