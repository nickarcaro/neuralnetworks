from flask import Flask, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

training_data = './dataset_SCL.csv'

def grafica_matriz_confunsion(labels, predictions, p=0.5):
  tick_labels = ['No Atraso', 'Atraso']

  cm = confusion_matrix(labels, predictions > p)
  ax = sns.heatmap(cm, annot=True, fmt="d")
  plt.ylabel('Actual')
  plt.xlabel('Predicción')
  ax.set_xticklabels(tick_labels)
  ax.set_yticklabels(tick_labels)

def plot_roc(labels, predictions):
  fp, tp, _ = roc_curve(labels, predictions)

  plt.plot(fp, tp, label='ROC', linewidth=3)
  plt.xlabel('Porcentage Falsos Positivos')
  plt.ylabel('Porcentaje veraderos positivos')
  plt.plot(
      [0, 1], [0, 1], 
      linestyle='--', 
      linewidth=2, 
      color='r',
      label='Chance', 
      alpha=.8
  )
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  plt.legend(loc="lower right")
app = Flask(__name__)

# route
@app.route('/predict', methods=['GET'])
def predict():
    

    df = pd.read_csv(training_data,low_memory=False)

    def temporada_alta(fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    df['temporada_alta'] = df['Fecha-I'].apply(temporada_alta)


    def dif_min(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        dif_min = ((fecha_o - fecha_i).total_seconds())/60
        return dif_min

    df['dif_min'] = df.apply(dif_min, axis = 1)
    df['atraso_15'] = np.where(df['dif_min'] > 15, 1, 0)



    def get_periodo_dia(fecha):
        fecha_time = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S').time()
        mañana_min = datetime.strptime("05:00", '%H:%M').time()
        mañana_max = datetime.strptime("11:59", '%H:%M').time()
        tarde_min = datetime.strptime("12:00", '%H:%M').time()
        tarde_max = datetime.strptime("18:59", '%H:%M').time()
        noche_min1 = datetime.strptime("19:00", '%H:%M').time()
        noche_max1 = datetime.strptime("23:59", '%H:%M').time()
        noche_min2 = datetime.strptime("00:00", '%H:%M').time()
        noche_max2 = datetime.strptime("4:59", '%H:%M').time()
        
        if(fecha_time > mañana_min and fecha_time < mañana_max):
            return 'mañana'
        elif(fecha_time > tarde_min and fecha_time < tarde_max):
            return 'tarde'
        elif((fecha_time > noche_min1 and fecha_time < noche_max1) or
            (fecha_time > noche_min2 and fecha_time < noche_max2)):
            return 'noche'

    df['periodo_dia'] = df['Fecha-I'].apply(get_periodo_dia)




    """Se utilizarán variables no usadas anteriormente para brindar más información relevante a los vuelos, en este caso factores de temporada alta y el período del día donde afectan directamente a los atrasos"""

    data_proposed = df[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'atraso_15','temporada_alta','periodo_dia']]

    """como se trabajará con árboles de decisión, en vez de trabajar con One-Hot encoding, se utilizará una codificación de etiquetas para asociar valores a los datos categoricos y disminuir la cantidad de features que genera One-hot encodig. Debido a que los arboles de decisión se sobre ajustan a un mayor numero de features."""

    #se codifica las clases 
    labelencoder = LabelEncoder()

    data_proposed['TIPOVUELO']= labelencoder.fit_transform(data_proposed['TIPOVUELO'])

    data_proposed['OPERA'] = labelencoder.fit_transform(data_proposed['OPERA'])

    data_proposed['DIANOM'] = labelencoder.fit_transform(data_proposed['DIANOM'])

    data_proposed['SIGLADES'] = labelencoder.fit_transform(data_proposed['SIGLADES'])
    data_proposed['periodo_dia'] = labelencoder.fit_transform(data_proposed['periodo_dia'])

    x = data_proposed[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM','temporada_alta','periodo_dia']]
    y = data_proposed['atraso_15']



    smote = SMOTE()

    # fit predictor and target variable
    x_rus, y_rus = smote.fit_resample(x, y)


    """#### Random Forest"""

    x_train3, x_test3, y_train3, y_test3 = train_test_split(x_rus, y_rus, test_size = 0.33, random_state = 42)

    rfc = RandomForestClassifier(class_weight="balanced")

    rfc = rfc.fit(x_train3,y_train3)
    y_pred3 = rfc.predict(x_test3)
    #print(confusion_matrix(y_test3, y_pred3))

    #print(classification_report(y_test3, y_pred3))

    #print(roc_curve(y_test3, y_pred3))
    
    return_message = 'confusion matrix: \t \n{0}, classification report: \t \n{1}'.format(confusion_matrix(y_test3, y_pred3), classification_report(y_test3, y_pred3)) 
    #return grafica_matriz_confunsion(y_test3, y_pred3)

    return return_message


    #return jsonify({'response': confusion_matrix(y_test3, y_pred3)}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)