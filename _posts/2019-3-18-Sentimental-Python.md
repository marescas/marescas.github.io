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

# Preproceso
Se ha decidido elaborar un preproceso del texto convirtiendo cada tweet a minúsculas además de sustituir las urls por la etiqueta url y las menciones por la etiqueta mention. Para ello se ha utilizado el siguiente script:
{% highlight python %}
url_re = re.compile("https?://[^\s]+")
mention_re = re.compile("@(\w+)")
def preprocessing(text):
    text_clean = url_re.sub("<url>",text)
    text_clean = mention_re.sub("<mention>", text_clean)
    text_clean = text_clean.lower()
    return text_clean 
{% endhighlight %}
    
# Representación 
Para explotar la información textual es necesario realizar una representación de la misma, para ello se ha realizado una representación basada en bolsa de palabras (Bag Of Words) con las siguientes características:
* n-gramas de carácteres (3-4-5).
* Ponderación utilizando TF-IDF.
* Eliminación de aquellos términos que aparecen en más del 95% de los documentos.
* Eliminación de aquellos términos que aparecen en un único documento.

La representación de los documentos ha sido posible gracias al uso de la librería scikit-learn instanciando un objeto de la clase TfidfVectorizer. El código utilizado es el siguiente:
{% highlight python %}
def representationBOW(corpus):
    vectorizerTrain = TfidfVectorizer(ngram_range = (3,5),max_df=0.95,min_df=2,analyzer="char_wb")
    bow = vectorizerTrain.fit_transform(corpus)
    return bow,vectorizerTrain
{% endhighlight %}
# Experimentación
Se ha optado por utilizar dos algoritmos: Support Vector Machines y Logistic Regression. Además, se ha utilizado exploración exhaustiva con el fin de obtener los mejores parámetros para cada algoritmo
## Support Vector Machines
Para las máquinas de vectores soporte se ha explorado distintas combinaciones de kernel y C con el fin de determinar cual produce mejores predicciones. El código utilizado es el siguiente:
{% highlight python %}
from sklearn.svm import SVC
kernels = ["rbf","linear"]
cs = [1,10,100,1000]
for kernel in kernels:
    for c in cs:
        clf = SVC(kernel=kernel,C=c)
        clf.fit(bowTrain, y_train)
        print("%s \t %d \t %.3f " %(kernel[0:3],c,clf.score(bowTest,y_test)))
{% endhighlight %}

## LogisticRegression
En cuanto al uso de la regresión logística se ha estudiado como interviene la inversa de la fuerza de regularización (C) en la disminución del error. El código utilizado es el siguiente:
{% highlight python %}
from sklearn.linear_model import LogisticRegression
cs = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 10, 100, 1000]
for c in cs:
    clf = LogisticRegression(C=c)
    clf.fit(bowTrain, y_train)
    print("%.3f \t %.3f " %(c,clf.score(bowTest,y_test)))
{% endhighlight %}

# Resultados
Los resultados son esperanzadores. Una vez realizada la exploración exhaustiva los mejores resultados obtenidos son los siguientes:
* Máquinas de vectores soporte con kernel linear y C = 1 consigue una precisión del 81.5%.
* Regresión Logística con C = 1 consigue una precision del 81.3%

# Conclusiones
Se ha desarrollado un sistema capaz de predecir el sentimiento del usuario al manifestar su opinión en Twitter. Para ello se ha utilizado un dataset con 14640 tweets sobre la opinión de los viajeros en Febrero de 2015. Además,la información textual ha sido procesada y representada utilizando técnicas de procesamiento de lenguaje natural para posteriormente realizar una tarea de aprendizaje.

Además de la aproximación propuesta se podría intentar realizar otras aproximaciones basadas por ejemplo en el uso de word embeddings como Word2Vec, de esta forma se podría capturar relaciones semánticas entre distintos términos y obtener una representación basada en las hipótesis de la semántica distribucional. Probablemente lo deje para una segunda versión...

El código completo está disponible en [GitHub](https://github.com/marescas/marescas.github.io/blob/master/SentimentalPython.ipynb)


