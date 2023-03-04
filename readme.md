# desafio ml enginer

consideraciones:

- Dentro de model-expose se encuentra una aplicación pequeña hecha en flask ppara montar el modelo escogido
- No poseo un computador unix (solo windows). Sin embargo, ejecuté wrk utilizando un micro distro de ubuntu usando WSL con los requisitos de la prueba.
- dentro del archivo to_expose_nicolas.ipynb explico el modelo que escogí y los cambios que realicé
- Para mejorar la métrica de la prueba, es necesario distribuir los procedimientos de procesar los datos y la predicción de modelo. Para esto se debiese utilizar un entorno en la nube (AWS, AZURE etc) donde la parte del aplicativo ( procesamiento de los datos) funcione fuera de la predicción y entrega de resultados. Ello conlleva a que al realizar la consulta mediante API rest, esta se enfoque en entregar el resultado y no hacer todo el procedimiento en el controlador de la consulta
