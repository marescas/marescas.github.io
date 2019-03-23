---
layout: post
comments: true
title: Detección de objetos utilizando redes neuronales profundas
---

En la actualidad, las redes neuronales profundas o Deep Learning ha permitido el desarrollo de nuevos algoritmos en el ámbito del reconocimiento de patrones. En particular, en obras de visión por computador, la redes neuronales convolucionales han fijado el estado del arte en tareas de segmentación o clasificación de imágenes.

En este post trataremos de construir un detector de objetos capaz de detectar hasta 1000 objetos distintos. Para ello, se utilizará el modelo VGG pre-entrenado sobre [Image-Net](http://www.image-net.org). Este modelo podría ser una simplificación del modelo usado por Facebook para describir las imágenes a usuarios invidentes [Facebook](https://www.theverge.com/2016/4/5/11364914/facebook-automatic-alt-tags-blind-visually-impared).

# VGG
VGG es un modelo de Deep Learning presentado por Karen Simonyan y Andrew Zisserman en el artículo [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf). Se trata de un modelo basado en redes convolucionales de 16 o 19 capas.

El modelo recibe como input una matriz de 224 x 224 x 3 a la que se le aplican un conjunto de capas ocultas que dan como resultado un vector de mil componentes con la probabilidad de que cada objeto este en la imagen. La arquitectura de la red se puede apreciar en la siguiente imagen:
![Arquitectura de la red](/images/vgg16.png)

El modelo presentado por Karen y Andrew fue entrenado durante 2 semanas sobre Image-Net. Una iniciativa para proporcionar a investigadores de todo el mundo una  base de datos de imágenes de fácil acceso. Actualmente cuenta con más de 14 millones de imágenes y más de 1 millón de imágenes anotadas.


# Keras
Para el desarrollo del detector es necesario el uso de la arquitectura VGG16 proporcionada por Keras. Keras es una librería de redes neuronales escrita en Python y de código abierto. En nuestro caso ejecutaremos Keras sobre Tensorflow.

Para ello primero es necesario instalar Tensorflow y todas sus dependencias. Para ello, la forma más cómoda es instalar [Anaconda](https://www.anaconda.com).

Una vez instalado Anaconda ya podemos instalar Keras. Tan sencillo como ejecutar desde el terminal  **pip install keras**.

# ¡A programar!
Ahora ya tenemos todos los ingredientes. Primero de todo vamos a importar algunas cosas que necesitaremos:
{% highlight python %}
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
{% endhighlight %}

Keras applications es un conjunto de modelos de Deep Learning preentrenados sobre grandes conjuntos de datos. ¡Nos vamos a evitar semanas de entrenamiento gracias a esto!

{% highlight python %}
modelo = VGG16()
{% endhighlight %}

La primera vez que lo ejecutemos descargará  la arquitectura y los pesos  de la red sobre ImageNet. Son aproximadamente 500 MB (puede tardar un rato...). Una vez descargada podemos imprimir la arquitectura de la red con la siguiente línea de código
{% highlight python %}
print(modelo.summary())
{% endhighlight %}

Vamos a utilizar la función load_img que nos proporciona Keras para cargar la imagen y escalarla al tamaño que buscamos para la red:
{% highlight python %}
imagen = load_img('test2.jpeg', target_size=(224, 224))
{% endhighlight %}

El código anterior nos devuelve un objeto de tipo PIL Image, para que la red pueda trabajar  necesita una matriz de Numpy en el formato que espera
{% highlight python %}
imagen = img_to_array(imagen)
imagen = imagen.reshape((1, imagen.shape[0], imagen.shape[1], imagen.shape[2])) #224*224*3 (RGB)
{% endhighlight %}

Una vez tenemos la entrada como queremos ya podemos pasársela a la red y ver que dice.
{% highlight python %}
predicciones = modelo.predict(imagen)
etiquetas = decode_predictions(predicciones)
{% endhighlight %}
De esta forma tenemos una lista con las etiquetas más probables. Podemos imprimir cada etiqueta con su respectiva probabilidad:
{% highlight python %}
for l in etiquetas[0]:
    print(l[1], l[2])
{% endhighlight %}

# Resultados
Se ha probado el algoritmo con varias imágenes y los resultados son acertados

![Maserati](/images/maserati.jpg)
**Resultados:**
sports_car 0.4424708
convertible 0.40847698
beach_wagon 0.039662655
car_wheel 0.024519864
racer 0.023201792

![Ninfa](/images/bird.jpg)
**Resultados**
ptarmigan 0.5499425
peacock 0.23322123
jay 0.0885262
sulphur-crested_cockatoo 0.06153025
hen 0.014728737

![Alan Turing](/images/turing.jpg)
**Resultados**
bow_tie 0.728437
military_uniform 0.08980917
suit 0.06565995
Windsor_tie 0.024184078
sunglasses 0.017918428

¡Aleluya! Detecta el maserati como un deportivo y la Ninfa como un tipo de ave. Algo curioso pasa con Turing, detecta la corbata pero no una persona.  
