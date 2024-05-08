# Importar librerías necesarias
from dash import Dash
import pandas as pd
from os import path, getcwd, system, listdir
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.graph_objs as go
import json
from ..modelamiento.mod_explicativo import *
from ..modelamiento.mod_predictivo import *
import plotly.graph_objects as go


try:
    import psycopg2
except:
    system("pip3 install psycopg2")
    import psycopg2

ruta_carpeta_proy = path.dirname(getcwd())
directorio_padre = ruta_carpeta_proy + '/actd_proyecto_2'


listdir()


#Cargar base de datos
connection = psycopg2.connect(
    dbname="datos_limpios",
    user="postgres",
    password="1234567890",
    host="proyecto2.cpgmg4am8kak.us-east-1.rds.amazonaws.com",
    port="5432"
)

#Funciones del dash

def read_dict_from_json(file_path):
    # Read the dictionary from a JSON file
    with open(file_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict

# Example usage:
imported_dict = read_dict_from_json(directorio_padre + '/tablero/feature_importance.json')
imported_dict

def plot_feature_importance(importances, highlight=['EDUCATION', 'AGE']):
    # Convert the dictionary to a list of tuples and sort it
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Extract features and their corresponding importances
    features = [feat for feat, imp in sorted_importances]
    values = [imp for feat, imp in sorted_importances]

    # Set colors and opacity based on highlight
    colors = ['rgba(0, 0, 255, 0.4)' if feature not in highlight else 'rgba(0, 0, 255, 1.0)' for feature in features]
    borders = ['white'] * len(features)  # White borders for all bars

    # Create the bar plot
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        opacity=0.8  # General opacity setting
    ))

    # Update layout
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white'
    )

    return fig

# Plotting the feature importance
feature_importance_plot = plot_feature_importance(imported_dict)

def importance_variables(imported_dict):
    # Extraer los datos de 'AGE' y 'EDUCATION' del diccionario
    age_data = imported_dict.get('AGE', [])
    education_data = imported_dict.get('EDUCATION', [])

    # Convertir los valores a formato de porcentaje con tres decimales
    age_percentage = "{:.3f}%".format(age_data * 100)
    education_percentage = "{:.3f}%".format(education_data * 100)

    # Crear DataFrame con los valores modificados
    df = pd.DataFrame({
        'AGE': [age_percentage],
        'EDUCATION': [education_percentage]
    })

    # Crear la tabla de importancia de variables desde el DataFrame
    importance_details = dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        style={'width': '100%', 'margin': 'auto'}
    )

    return importance_details

variables = importance_variables(imported_dict)


#Definicion del Layout



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

sidebar = html.Div(
    [
        html.H1("Analítica Riesgo"),
        dbc.Nav(
            [
                dbc.NavLink("Información General", href="/info", active="exact"),
                dbc.NavLink("Explicativo", href="/explic", active="exact"),
                dbc.NavLink("Predictivo", href="/predict", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"width": "20%", "position": "fixed", "height": "100%", "background-color": "#f8f9fa"}
)

content = html.Div(id="page-content", style={"margin-left": "20%", "width": "80%"})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content,
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/info":
        return html.Div([
            html.H1("Información General"),
            html.H2("Pregunta Descriptiva:"),
            html.P("¿Existe una importancia significativa entre la edad, nivel educativo de los clientes y su propensión al default en los últimos meses?"),
            html.H2("Pregunta Predictiva:"),
            html.P("¿Cuál es la probabilidad de default en clientes de tarjetas de crédito, considerando variables clave como el historial de pagos, el estado de la cuenta bancaria, la cantidad de pagos anteriores y el monto de crédito otorgado, mientras se considera la relación temporal de los pagos y la edad del cliente?"),
        ])
    elif pathname == "/explic":
        return html.Div([
            
            html.Br(),
            html.Hr(),
            html.H1("Análisis Explicativo"),

            html.Br(),
            html.Hr(),
            html.H4("Indice de Importancia por Categorias"),
            
            # Insert the Plotly graph here
            html.Div([
                dcc.Graph(
                    id='feature-importance-graph',
                    figure=feature_importance_plot  # This assumes your Plotly figure is named 'feature_importance_plot'
                )
            ], id='feature_importance'),

            html.Br(),
            html.Hr(),
            html.H4("Variables Edad y Nivel Educativo"),
            html.Table(id='importance-variables'),
            
            # Inserta la tabla de importance_details aquí
            html.Div([
                html.P("Valores Importancia"),
                variables
            ])
            
        ])
    elif pathname == "/predict":
        return html.Div([

            html.Br(),
            html.Hr(),
            html.H4("Probabilidad de Incumplimiento de Pago"),
            html.Label("Seleccionar ID del Cliente"),
            dcc.Input(id='customer-id', type='number', value=12),

            html.Br(),
            html.Hr(),
            html.Div(id='default-probability'),
            html.H4("Detalles del Cliente"),
            html.Table(id='customer-details'),
            
            html.Br(),
            html.Hr(),
            html.H4("Monto de Crédito Otorgado"),
            html.Table(id='credit-limit'),

            html.Br(),
            html.Hr(),
            html.H4("Estado de Cuenta y Montos Pagados a lo largo del Tiempo"),
            dcc.Graph(id='line-chart')

            
        ])
    else:
        return "404 Page Not Found"


# Callbacks para actualizar los componentes del dashboard
@app.callback(
    Output('default-probability', 'children'),
    Output('customer-details', 'children'),
    Output('line-chart', 'figure'),
    Output('credit-limit', 'children'),
    Input('customer-id', 'value')
)
def update_customer_info(customer_id):


    # Consultar la base de datos para obtener los detalles del cliente
    query = f"SELECT sex, education, marriage, age, limit_bal FROM datos_limpios WHERE id = {customer_id}"
    customer_details = pd.read_sql_query(query, connection)
    
    # Mapear los valores numéricos a sus respectivas categorías
    sex_mapping = {1: 'Masculino', 2: 'Femenino'}
    education_mapping = {1: 'Escuela de Graduados', 2: 'Universidad', 3: 'Escuela Secundaria', 4: 'Otros'}
    marriage_mapping = {1: 'Casado', 2: 'Soltero', 3: 'Otros'}

    # Aplicar el mapeo a las columnas correspondientes
    customer_details['sex'] = customer_details['sex'].map(sex_mapping)
    customer_details['education'] = customer_details['education'].map(education_mapping)
    customer_details['marriage'] = customer_details['marriage'].map(marriage_mapping)

    # Modificar el nombre de las variables en el DataFrame
    customer_details.rename(columns={'sex': 'Sexo', 'education': 'Nivel Educativo', 'marriage': 'Estado Civil','age': 'Edad', 'limit_bal': 'Monto Crédito'}, inplace=True)
    
    # Example database query and processing logic
    # Assuming the function proba_dado_id() returns a probability as a float
    default_probability = round(proba_dado_id(customer_id) * 100, 3)

    # Create gauge plot for the default probability
    gauge_figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=default_probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Probability"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': 'lightgreen'},
                   {'range': [50, 75], 'color': 'yellow'},
                   {'range': [75, 100], 'color': 'red'},
               ],
               'threshold': {
                   'line': {'color': "black", 'width': 4},
                   'thickness': 0.75,
                   'value': default_probability}
        }
    ))

    gauge_plot = dcc.Graph(figure=gauge_figure)

    # Actualizar el número con la probabilidad de incumplimiento de pago
    default_prob_component = [html.H2(f"Probabilidad de Incumplimiento de Pago: {default_probability}%"), gauge_plot ]

    # Actualizar componente con detalles del cliente
    customer_details_component = dbc.Table.from_dataframe(
        customer_details[['Sexo', 'Nivel Educativo',  'Estado Civil', 'Edad']],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        style={'width': '100%', 'margin': 'auto'}
    )
    
    # Actualizar tabla con el monto de crédito otorgado
    credit_limit_component = html.Div([
        dbc.Table.from_dataframe(
            pd.DataFrame({'Monto de Crédito Otorgado': [customer_details['Monto Crédito'][0]]}),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            style={'width': '100%', 'margin': 'auto'}
        )
    ])

    # Consultar la base de datos para obtener los datos de bill_amt y pay_amt
    query_bill_amt = f"SELECT bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6 FROM datos_limpios WHERE id = {customer_id}"
    query_pay_amt = f"SELECT pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6 FROM datos_limpios WHERE id = {customer_id}"
    bill_amt_data = pd.read_sql_query(query_bill_amt, connection)
    pay_amt_data = pd.read_sql_query(query_pay_amt, connection)

    # Crear el DataFrame con las columnas solicitadas
    df = {
        'Meses': ['Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre'],
        'Bill Amount': bill_amt_data.values.flatten(),
        'Pay Amount': pay_amt_data.values.flatten()
    }

    # Crear el gráfico de línea para Bill_amt y Pay_amt
    line_plot_figure = go.Figure()
    line_plot_figure.add_trace(go.Scatter(x=df['Meses'], y=df['Bill Amount'], mode='lines', name='Bill Amount', line=dict(width=2)))
    line_plot_figure.add_trace(go.Scatter(x=df['Meses'], y=df['Pay Amount'], mode='lines', name='Pay Amount', line=dict(width=2)))

    # Actualizar el diseño del gráfico
    line_plot_figure.update_layout(
        margin=dict(t=30, l=30, r=30, b=30),  # Margins
        legend=dict(x=0.75, y=1, bgcolor='rgba(255, 255, 255, 0.5)'),  # Legend
        xaxis_title="Meses",
        yaxis_title="Valor"
    )

    # Ajustar el tamaño de los puntos
    line_plot_figure.update_traces(marker=dict(size=12, opacity=0.9))



    return default_prob_component, customer_details_component,  line_plot_figure, credit_limit_component

#Correr la app



port = 8050

if __name__ == '__main__':
    url = f"http://127.0.0.1:{port}"
    print(f"Dash app running on {url}")
    app.run_server(debug=False, port=port)
    