import os
import dill
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["DEBUG"] = True

# Definir rutas globales para el modelo
BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
MODEL_DIR = os.path.join(BASE_PATH, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        pipeline = dill.load(f)
    return pipeline

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Bienvenido a  nuestra api del modelo 'Airline Passenger Satisfaction'</h1>
    <p>Este espacio es un prototipo de API.</p>
    <p> Para más información sobre el modelo, <a href="https://github.com/Marpp98/Pipelines_Airline_Passenger_Satisfaction" target="_blank">pincha aquí</a>.</p>
    """


@app.route("/param_info/<param>")
def param_info(param):
    """
    Muestra información adicional acerca del parámetro.
    """
    descriptions = {
        "Gender": "El género del pasajero (por ejemplo, 'Male' o 'Female').",
        "Customer Type": "El tipo de cliente (por ejemplo, 'Loyal' o 'disloyal').",
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
      <head><title>Información de {param}</title></head>
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
            # Recoger los datos enviados del formulario
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
            # Convertir a DataFrame
            df_input = pd.DataFrame([input_data])
            # Agregar la columna dummy 'satisfaction' para mantener la consistencia
            df_input["satisfaction"] = "neutral or dissatisfied"
            
            # Cargar el pipeline desde el archivo .pkl
            pipeline = load_pipeline()
            y_pred = pipeline.predict(df_input)
            pred_value = int(y_pred[0])
            pred_label = "satisfied" if pred_value == 1 else "neutral or dissatisfied"
            
            return f"<h1>Resultado: {pred_label}</h1><br><a href='/predict'>Volver al formulario</a>"
        except Exception as e:
            return f"<h1>Error: {str(e)}</h1><br><a href='/predict'>Volver al formulario</a>"
    else:
        # HTML del formulario con links de información al lado de cada parámetro.
        form_html = """
        <html>
          <head>
            <title>Formulario de Predicción</title>
          </head>
          <body>
            <h1>Formulario de Predicción</h1>
            <p>Para más información sobre un parámetro, pincha en el enlace [info] a su lado.</p>
            <form method="post" action="/predict">
              <label>Gender:</label>
              <input type="text" name="Gender" required>
              <a href="/param_info/Gender" target="_blank">[info]</a><br><br>
              
              <label>Customer Type:</label>
              <input type="text" name="Customer Type" required>
              <a href="/param_info/Customer Type" target="_blank">[info]</a><br><br>
              
              <label>Age:</label>
              <input type="number" step="any" name="Age" required>
              <a href="/param_info/Age" target="_blank">[info]</a><br><br>
              
              <label>Type of Travel:</label>
              <input type="text" name="Type of Travel" required>
              <a href="/param_info/Type of Travel" target="_blank">[info]</a><br><br>
              
              <label>Class:</label>
              <input type="text" name="Class" required>
              <a href="/param_info/Class" target="_blank">[info]</a><br><br>
              
              <label>Flight Distance:</label>
              <input type="number" step="any" name="Flight Distance" required>
              <a href="/param_info/Flight Distance" target="_blank">[info]</a><br><br>
              
              <label>Inflight wifi service:</label>
              <input type="number" step="any" name="Inflight wifi service" required>
              <a href="/param_info/Inflight wifi service" target="_blank">[info]</a><br><br>
              
              <label>Departure/Arrival time convenient:</label>
              <input type="number" step="any" name="Departure/Arrival time convenient" required>
              <a href="/param_info/Departure/Arrival time convenient" target="_blank">[info]</a><br><br>
              
              <label>Ease of Online booking:</label>
              <input type="number" step="any" name="Ease of Online booking" required>
              <a href="/param_info/Ease of Online booking" target="_blank">[info]</a><br><br>
              
              <label>Gate location:</label>
              <input type="number" step="any" name="Gate location" required>
              <a href="/param_info/Gate location" target="_blank">[info]</a><br><br>
              
              <label>Food and drink:</label>
              <input type="number" step="any" name="Food and drink" required>
              <a href="/param_info/Food and drink" target="_blank">[info]</a><br><br>
              
              <label>Online boarding:</label>
              <input type="number" step="any" name="Online boarding" required>
              <a href="/param_info/Online boarding" target="_blank">[info]</a><br><br>
              
              <label>Seat comfort:</label>
              <input type="number" step="any" name="Seat comfort" required>
              <a href="/param_info/Seat comfort" target="_blank">[info]</a><br><br>
              
              <label>Inflight entertainment:</label>
              <input type="number" step="any" name="Inflight entertainment" required>
              <a href="/param_info/Inflight entertainment" target="_blank">[info]</a><br><br>
              
              <label>On-board service:</label>
              <input type="number" step="any" name="On-board service" required>
              <a href="/param_info/On-board service" target="_blank">[info]</a><br><br>
              
              <label>Leg room service:</label>
              <input type="number" step="any" name="Leg room service" required>
              <a href="/param_info/Leg room service" target="_blank">[info]</a><br><br>
              
              <label>Baggage handling:</label>
              <input type="number" step="any" name="Baggage handling" required>
              <a href="/param_info/Baggage handling" target="_blank">[info]</a><br><br>
              
              <label>Checkin service:</label>
              <input type="number" step="any" name="Checkin service" required>
              <a href="/param_info/Checkin service" target="_blank">[info]</a><br><br>
              
              <label>Inflight service:</label>
              <input type="number" step="any" name="Inflight service" required>
              <a href="/param_info/Inflight service" target="_blank">[info]</a><br><br>
              
              <label>Cleanliness:</label>
              <input type="number" step="any" name="Cleanliness" required>
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


# Endpoint Status: devuelve información sobre el estado de la API y el modelo.
@app.route("/status", methods=["GET"])
def status():
    pipeline = load_pipeline()
    # Extraer información básica:
    model_step = pipeline.named_steps.get("model")
    if model_step is not None and hasattr(model_step, "named_steps"):
        # Suponiendo que el modelo real está dentro de la etapa "model" del pipeline anidado.
        best_model = model_step.named_steps.get("model")
        model_type = best_model.__class__.__name__ if best_model is not None else "Desconocido"
    else:
        model_type = "Desconocido"
    
    # Se listan los pasos principales del pipeline para ofrecer una visión global.
    steps = [name for name, _ in pipeline.steps]
    
    status_info = {
        "status": "OK",
        "model_type": model_type,
        "pipeline_steps": steps,
        "message": "La API está funcionando correctamente."
    }
    
    return jsonify(status_info)

if __name__ == "__main__":
    app.run(debug=True)
