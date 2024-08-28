import numpy as np, pandas as pd 
import streamlit as st
import pickle
import joblib

#knn = pickle.load(open('knn_model.pkl','rb'))
knn = joblib.load('knn_model.pkl')

st.markdown("<h1  style='text-align: center;'> <u> Car Price Prediction Prompt </u></h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Fill The Details</h2>", unsafe_allow_html=True)

options = st.sidebar.selectbox('Select the Model',['KNN'])

brand_to_models = {
    'Maruti': ['800', 'Wagon', 'Swift', 'Eeco', 'Vitara', 'Baleno', 'Celerio', 'Omni', 'Zen', 'Esteem', 'Alto', 'A-Star', 'Ritz', 'SX4', 'S-Cross', 'Ertiga', 'Ciaz', 'Ignis', 'S-Presso'],
    'Hyundai': ['Verna', 'Creta', 'i10', 'i20', 'Elite', 'Grand', 'Venue', 'Santro', 'Xcent', 'EON', 'Getz', 'Elantra', 'Sonata', 'Tucson', 'Accent'],
    'Others': ['Jeep', 'Q5', 'Q7', 'X5', 'BR-V', 'XUV500', 'XUV300', 'Fortuner', 'Pajero', 'Hector', 'Seltos', 'BRV', 'Rover', 'Ingenio', 'GL-Class', 'XC60'],
    'Honda': ['City', 'Amaze', 'Civic', 'Jazz', 'BR-V', 'Accord', 'CR-V', 'WR-V'],
    'Tata': ['Indigo', 'Nano', 'Sumo', 'Safari', 'Tiago', 'Nexon', 'Harrier', 'Tigor', 'Altroz', 'Hexa', 'Aria', 'Zest', 'Bolt', 'Manza', 'Indica'],
    'Chevrolet': ['Sail', 'Enjoy', 'Cruze', 'Beat', 'Spark', 'Captiva', 'Aveo', 'Optra'],
    'Toyota': ['Corolla', 'Innova', 'Fortuner', 'Etios', 'Yaris', 'Camry', 'Qualis'],
    'Mahindra': ['Scorpio', 'XUV500', 'XUV300', 'Bolero', 'Marazzo', 'Thar', 'TUV', 'Alturas', 'KUV'],
    'Ford': ['Ecosport', 'Figo', 'Fiesta', 'Endeavour', 'Aspire', 'Freestyle', 'Ikon'],
    'Renault': ['Kwid', 'Duster', 'Scala', 'Pulse', 'Fluence', 'Koleos', 'Triber', 'Captur', 'Lodgy']
}


name = st.selectbox('Brand',list(brand_to_models.keys()))

if name == 'Maruti':
    name_Maruti = 1
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0
elif name == 'Hyundai':
    name_Maruti = 0
    name_Hyundai = 1
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0
elif name == 'Others':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 1
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0
elif name == 'Honda':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 1
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0
elif name == 'Tata':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 1
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0
elif name == 'Renault':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 1
elif name == 'Toyota':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 1  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0
elif name == 'Mahindra':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 1
    name_Ford = 0
    name_Renault = 0
elif name == 'Ford':
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 0 
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 1
    name_Renault = 0
else:
    name_Maruti = 0
    name_Hyundai = 0
    name_Others = 0
    name_Honda = 0
    name_Tata = 0
    name_Chevrolet = 1
    name_Toyota = 0  
    name_Mahindra = 0
    name_Ford = 0
    name_Renault = 0



year = st.selectbox('Made In Year',list(range(1990, 2021)))


km_driven = st.slider('Driven In KM',1,900000)
yeo_ = joblib.load('power_transformer_model.pkl')

km_driven_1 = np.array([[km_driven]])
km_driven = yeo_.transform(km_driven_1).flatten()[0]


model = st.selectbox('Model',brand_to_models[name])
_mean = joblib.load('target_encoding_model.pkl')
model_1 = np.array([[model]])
mean_map = np.vectorize(_mean.get)
model_2 = mean_map(model_1)
model = yeo_.transform(model_2).flatten()[0]


fuel = st.selectbox('Fuel Type',['Petrol', 'Diesel', 'Others'])

if fuel == 'Petrol':
    fuel_Petrol = 1
    fuel_Diesel = 0
    fuel_Others = 0
elif fuel == 'Others':
    fuel_Petrol = 0
    fuel_Diesel = 0 
    fuel_Others = 1
else:
    fuel_Petrol = 0
    fuel_Diesel = 1
    fuel_Others = 0

transmission = st.selectbox('Transmission',['Manual', 'Automatic'])
if transmission == 'Manual':
    transmission_Manual = 1
    transmission_Automatic = 0
else:
    transmission_Manual = 0
    transmission_Automatic = 1

owner = st.selectbox('Owner Type',['First Owner', 'Second Owner', 'Third & Above Owner'])

if owner == 'First Owner':
    owner = 0
elif owner == 'Second Owner':
     owner = 1
else:
    owner = 2
    


test = [year, km_driven, owner,model,name_Ford,name_Honda,name_Hyundai, name_Mahindra, name_Maruti, name_Others ,name_Renault,
         name_Tata, name_Toyota, fuel_Others, fuel_Petrol, transmission_Manual]


rob_ = joblib.load('RobustScaler_model.pkl')
test_0 = np.array(test).reshape(1,16)
test_1 = rob_.transform(test_0)


if st.button('Predict'):
    if options =="KNN":
        st.success(f" Predicted car price is ₹ {knn.predict(test_1)[0]}")


# to run terminal >>> streamlit run app1.py
