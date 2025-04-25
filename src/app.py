import os
import sys
import dill
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["DEBUG"] = True

# Definir rutas con ruta absoluta
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "model.pkl")

# Definir la variable global con las columnas numéricas a transformar (según Notebook I)
numerical_cols_log = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

def safe_log_transform_fixed(X):
    if not isinstance(X, pd.DataFrame):
        try:
            X = pd.DataFrame(X)
        except Exception:
            return X
    if isinstance(X, pd.DataFrame):
        X_copy = X.copy()
        for col in numerical_cols_log:
            try:
                X_copy[col] = pd.to_numeric(X_copy[col], errors="coerce")
                X_copy[col] = np.log1p(X_copy[col])
            except Exception:
                pass
        return X_copy
    else:
        try:
            arr = np.array(X)
            return np.log1p(arr.astype(float))
        except Exception:
            return X

# Monkey-patching para que el pipeline use la función corregida
sys.modules["__main__"].safe_log_transform = safe_log_transform_fixed

def load_pipeline():
    import pandas as pd
    import numpy as np
    import __main__
    __main__.pd = pd
    __main__.np = np

    __main__.numerical_cols_log = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    __main__.safe_log_transform = safe_log_transform_fixed

    with open(MODEL_PATH, "rb") as f:
        pipeline = dill.load(f)
    return pipeline

@app.route('/', methods=['GET'])
def home():
    # Lista de parámetros disponibles para ver info
    params = [
        "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness", "Departure Delay in Minutes",
        "Arrival Delay in Minutes"
    ]
    home_html = f"""
    <html>
      <head>
        <title>Inicio - Airline Passenger Satisfaction API</title>
        <script>
          function openParamInfo() {{
            var e = document.getElementById("paramsDropdown");
            var selectedParam = e.options[e.selectedIndex].value;
            var url = "/param_info/" + encodeURIComponent(selectedParam);
            window.open(url, "_blank");
          }}
        </script>
      </head>
      <body>
        <h1>Bienvenido a nuestra API del modelo "Airline Passenger Satisfaction"</h1>
        <p>Este espacio es un prototipo de API.</p>
        <p>Para más información sobre el modelo, <a href="https://github.com/Marpp98/Pipelines_Airline_Passenger_Satisfaction" target="_blank">pincha aquí</a>.</p>
        <h2>Endpoints disponibles</h2>
        <ul>
          <li><a href="/predict" target="_blank">Predict</a></li>
          <li><a href="/status" target="_blank">Status</a></li>
          <li>
            <span>Información sobre los parámetros:</span>
            <select id="paramsDropdown">
              {''.join([f"<option value='{p}'>{p}</option>" for p in params])}
            </select>
            <button onclick="openParamInfo()">Saber más</button>
          </li>

        </ul>
      </body>
    </html>
    """
    return home_html

@app.route("/param_info/<param>")
def param_info(param):
    descriptions = {
        "Gender": "El género del pasajero (por ejemplo, 'Male' o 'Female').",
        "Customer Type": "El tipo de cliente (por ejemplo, 'Loyal' o 'Disloyal').",
        "Age": "La edad del pasajero en años.",
        "Type of Travel": "El tipo de viaje (por ejemplo, 'Business' o 'Personal').",
        "Class": "La clase del vuelo (por ejemplo, 'Economy', 'Business').",
        "Flight Distance": "La distancia del vuelo en kilómetros.",
        "Inflight wifi service": "La calificación del servicio de wifi a bordo (1-5).",
        "Departure/Arrival time convenient": "La conveniencia de la hora de salida/llegada (1-5).",
        "Ease of Online booking": "La facilidad de reservar en línea (1-5).",
        "Gate location": "La posición de la puerta de embarque (1-5).",
        "Food and drink": "La calificación de alimentos y bebidas (1-5).",
        "Online boarding": "La calificación del embarque en línea (1-5).",
        "Seat comfort": "La comodidad del asiento (1-5).",
        "Inflight entertainment": "La calificación del entretenimiento a bordo (1-5).",
        "On-board service": "La calificación del servicio a bordo (1-5).",
        "Leg room service": "La calificación del espacio para las piernas (1-5).",
        "Baggage handling": "La calificación del manejo de equipaje (1-5).",
        "Checkin service": "La calificación del servicio de checkin (1-5).",
        "Inflight service": "La calificación del servicio a bordo (1-5).",
        "Cleanliness": "La calificación de la limpieza (1-5).",
        "Departure Delay in Minutes": "El retraso de salida en minutos.",
        "Arrival Delay in Minutes": "El retraso de llegada en minutos"
    }
    info = descriptions.get(param, "No se tiene información específica para este parámetro.")
    html_info = f"""
    <html>
      <head>
        <title>Información de {param}</title>
      </head>
      <body>
        <h1>Información sobre {param}</h1>
        <p>{info}</p>
        <a href="/predict">Volver al formulario</a>
      </body>
    </html>
    """
    return html_info

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            input_data = {
                "Gender": request.form.get("Gender"),
                "Customer Type": request.form.get("Customer Type"),
                "Age": float(request.form.get("Age")),
                "Type of Travel": request.form.get("Type of Travel"),
                "Class": request.form.get("Class"),
                "Flight Distance": float(request.form.get("Flight Distance")),
                "Inflight wifi service": float(request.form.get("Inflight wifi service")),
                "Departure/Arrival time convenient": float(request.form.get("Departure/Arrival time convenient")),
                "Ease of Online booking": float(request.form.get("Ease of Online booking")),
                "Gate location": float(request.form.get("Gate location")),
                "Food and drink": float(request.form.get("Food and drink")),
                "Online boarding": float(request.form.get("Online boarding")),
                "Seat comfort": float(request.form.get("Seat comfort")),
                "Inflight entertainment": float(request.form.get("Inflight entertainment")),
                "On-board service": float(request.form.get("On-board service")),
                "Leg room service": float(request.form.get("Leg room service")),
                "Baggage handling": float(request.form.get("Baggage handling")),
                "Checkin service": float(request.form.get("Checkin service")),
                "Inflight service": float(request.form.get("Inflight service")),
                "Cleanliness": float(request.form.get("Cleanliness")),
                "Departure Delay in Minutes": float(request.form.get("Departure Delay in Minutes")),
                "Arrival Delay in Minutes": float(request.form.get("Arrival Delay in Minutes"))
            }
            df_input = pd.DataFrame([input_data])
            df_input["satisfaction"] = "neutral or dissatisfied"
            
            pipeline = load_pipeline()
            transformed = pipeline[:-1].transform(df_input)
            X_final = transformed[:, 1:]
            y_pred = pipeline.named_steps["model"].predict(X_final)
            pred_value = int(y_pred[0])
            pred_label = "satisfied" if pred_value == 1 else "neutral or dissatisfied"
            
            return f"<h1>Resultado: {pred_label}</h1><br><a href='/predict'>Volver al formulario</a>"
        except Exception as e:
            return f"<h1>Error: {str(e)}</h1><br><a href='/predict'>Volver al formulario</a>"
    else:
        gender_options = ['Male', 'Female']
        customer_type_options = ['Loyal', 'Disloyal']
        travel_type_options = ['Business', 'Personal']
        class_options = ['Economy', 'Business']
        calificaciones_1_5 = [1, 2, 3, 4, 5]
        
        form_html = f"""
        <html>
          <head>
            <title>Formulario de Predicción</title>
          </head>
          <body>
            <h1>Formulario de Predicción</h1>
            <p>Para más información sobre un parámetro, pincha en el enlace [info] a su lado.</p>
            <form method="post" action="/predict">
              
              <label>Gender:</label>
              <select name="Gender" required>
                {''.join([f"<option value='{opt}'>{opt}</option>" for opt in gender_options])}
              </select>
              <a href="/param_info/Gender" target="_blank">[info]</a><br><br>
              
              <label>Customer Type:</label>
              <select name="Customer Type" required>
                {''.join([f"<option value='{opt}'>{opt}</option>" for opt in customer_type_options])}
              </select>
              <a href="/param_info/Customer Type" target="_blank">[info]</a><br><br>
              
              <label>Age:</label>
              <input type="number" step="any" name="Age" required>
              <a href="/param_info/Age" target="_blank">[info]</a><br><br>
              
              <label>Type of Travel:</label>
              <select name="Type of Travel" required>
                {''.join([f"<option value='{opt}'>{opt}</option>" for opt in travel_type_options])}
              </select>
              <a href="/param_info/Type of Travel" target="_blank">[info]</a><br><br>
              
              <label>Class:</label>
              <select name="Class" required>
                {''.join([f"<option value='{opt}'>{opt}</option>" for opt in class_options])}
              </select>
              <a href="/param_info/Class" target="_blank">[info]</a><br><br>
              
              <label>Flight Distance:</label>
              <input type="number" step="any" name="Flight Distance" required>
              <a href="/param_info/Flight Distance" target="_blank">[info]</a><br><br>
              
              <label>Inflight wifi service:</label>
              <select name="Inflight wifi service" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Inflight wifi service" target="_blank">[info]</a><br><br>
              
              <label>Departure/Arrival time convenient:</label>
              <select name="Departure/Arrival time convenient" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Departure/Arrival time convenient" target="_blank">[info]</a><br><br>
              
              <label>Ease of Online booking:</label>
              <select name="Ease of Online booking" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Ease of Online booking" target="_blank">[info]</a><br><br>
              
              <label>Gate location:</label>
              <select name="Gate location" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Gate location" target="_blank">[info]</a><br><br>
              
              <label>Food and drink:</label>
              <select name="Food and drink" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Food and drink" target="_blank">[info]</a><br><br>
              
              <label>Online boarding:</label>
              <select name="Online boarding" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Online boarding" target="_blank">[info]</a><br><br>
              
              <label>Seat comfort:</label>
              <select name="Seat comfort" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Seat comfort" target="_blank">[info]</a><br><br>
              
              <label>Inflight entertainment:</label>
              <select name="Inflight entertainment" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Inflight entertainment" target="_blank">[info]</a><br><br>
              
              <label>On-board service:</label>
              <select name="On-board service" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/On-board service" target="_blank">[info]</a><br><br>
              
              <label>Leg room service:</label>
              <select name="Leg room service" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Leg room service" target="_blank">[info]</a><br><br>
              
              <label>Baggage handling:</label>
              <select name="Baggage handling" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Baggage handling" target="_blank">[info]</a><br><br>
              
              <label>Checkin service:</label>
              <select name="Checkin service" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Checkin service" target="_blank">[info]</a><br><br>
              
              <label>Inflight service:</label>
              <select name="Inflight service" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Inflight service" target="_blank">[info]</a><br><br>
              
              <label>Cleanliness:</label>
              <select name="Cleanliness" required>
                {''.join([f"<option value='{c}'>{c}</option>" for c in calificaciones_1_5])}
              </select>
              <a href="/param_info/Cleanliness" target="_blank">[info]</a><br><br>
              
              <label>Departure Delay in Minutes:</label>
              <input type="number" step="any" name="Departure Delay in Minutes" required>
              <a href="/param_info/Departure Delay in Minutes" target="_blank">[info]</a><br><br>
              
              <label>Arrival Delay in Minutes:</label>
              <input type="number" step="any" name="Arrival Delay in Minutes" required>
              <a href="/param_info/Arrival Delay in Minutes" target="_blank">[info]</a><br><br>
              
              <button type="submit">Predecir</button>
            </form>
          </body>
        </html>
         """
        return render_template_string(form_html)

@app.route("/status", methods=["GET"])
def status():
    pipeline = load_pipeline()
    model_step = pipeline.named_steps.get("model")
    if model_step is not None and hasattr(model_step, "named_steps"):
        best_model = model_step.named_steps.get("model")
        model_type = best_model.__class__.__name__ if best_model is not None else "Desconocido"
    else:
        model_type = "Desconocido"
    
    steps = [name for name, _ in pipeline.steps]
    
    status_info = {
        "status": "OK",
        "model_type": model_type,
        "pipeline_steps": steps,
        "message": "La API funciona correctamente."
    }
    
    return jsonify(status_info)

if __name__ == "__main__":
    app.run(debug=True)
