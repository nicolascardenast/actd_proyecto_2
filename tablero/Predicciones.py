# Importar librerías necesarias
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import psycopg2
from dotenv import load_dotenv 
import os
import dash_bootstrap_components as dbc
import os
import re
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import random

from modelamiento import proba_dado_id


# Cargar variables de entorno
load_dotenv()

# Conexión a la base de datos
connection = psycopg2.connect(
    dbname="datos_limpios",
    user="postgres",
    password="1234567890",
    host="proyecto2.cpgmg4am8kak.us-east-1.rds.amazonaws.com",
    port="5432"
)
# Crear aplicación Dash
app = dash.Dash(__name__)

# Definir el diseño del dashboard
app.layout = html.Div([
    html.H1("Probabilidad de Incumplimiento de Pago"),

    # Selección del ID del cliente
    html.Label("Seleccionar ID del Cliente"),
    dcc.Input(id='customer-id', type='number', value=12),

    # Mostrar probabilidad de incumplimiento de pago
    html.Div(id='default-probability'),

    # Tabla con detalles del cliente
    html.H2("Detalles del Cliente"),
    html.Table(id='customer-details'),

    # Gráfico de líneas
    html.H2("Estado de Cuenta y Montos Pagados a lo largo del Tiempo"),
    dcc.Graph(id='line-chart'),

    # Tabla con monto de crédito otorgado
    html.H2("Monto de Crédito Otorgado"),
    html.Table(id='credit-limit')
])

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

    # Calcular la probabilidad de incumplimiento de pago
    default_probability = proba_dado_id(customer_id)

    # Actualizar el número grande con la probabilidad de incumplimiento de pago
    default_prob_component = html.H2(f"Probabilidad de Incumplimiento de Pago: {default_probability}")

    # Actualizar la tabla de detalles del cliente
    customer_details_component = html.Table([
        html.Tr([html.Th(col.capitalize().replace('_', ' ')) for col in customer_details.columns]),
        html.Tr([html.Td(customer_details.iloc[0][col]) for col in customer_details.columns])
    ])

    # Consultar la base de datos para obtener los datos de bill_amt y pay_amt
    # Consultar la base de datos para obtener los datos de bill_amt y pay_amt
    query_bill_amt = f"SELECT bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6 FROM datos_limpios WHERE id = {customer_id}"
    query_pay_amt = f"SELECT pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6 FROM datos_limpios WHERE id = {customer_id}"
    bill_amt_data = pd.read_sql_query(query_bill_amt, connection)
    pay_amt_data = pd.read_sql_query(query_pay_amt, connection)

    # Crear el DataFrame con las columnas solicitadas
    df = {
        'Meses': ['Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre'],
        'Bill_amt': bill_amt_data.values.flatten(),
        'Pay_amt': pay_amt_data.values.flatten()
}
    # Crear el gráfico de dispersión para Bill_amt y Pay_amt
    scatter_plot_figure = go.Figure()
    scatter_plot_figure.add_trace(go.Scatter(x=df['Meses'], y=df['Bill_amt'], mode='markers', name='Bill_amt'))
    scatter_plot_figure.add_trace(go.Scatter(x=df['Meses'], y=df['Pay_amt'], mode='markers', name='Pay_amt'))
    scatter_plot_figure.update_layout(title='Scatter Plot de Meses vs Valores de Bill_amt y Pay_amt', xaxis_title='Meses', yaxis_title='Valor')
    # Actualizar la tabla con el monto de crédito otorgado
    credit_limit_component = html.Table([
        html.Tr([html.Th("Monto de Crédito Otorgado")]),
        html.Tr([html.Td(customer_details['limit_bal'][0])])
    ])

    return default_prob_component, customer_details_component, scatter_plot_figure, credit_limit_component

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)