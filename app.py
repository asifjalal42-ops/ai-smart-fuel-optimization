import os
import json
import sqlite3
import random
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "fuel_database.db"
DATA_PATH = APP_DIR / "fuel_training_data.csv"

load_dotenv(APP_DIR / ".env")

st.set_page_config(
    page_title="Smart Fuel AI System",
    page_icon="⛽",
    layout="wide"
)

st.title("⛽ Smart Fuel Consumption Monitoring System")

st.markdown(
    """
    **Mechanical Engineering | Artificial Intelligence |
    Agentic Artificial Intelligence**

    An intelligent platform for vehicle fuel monitoring,
    consumption prediction and fuel-budget optimization.
    """
)

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.5"
)


# ============================================================
# 2. DATABASE INITIALIZATION
# ============================================================

def get_connection():

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            efficiency REAL NOT NULL,
            tank_capacity REAL NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fuel_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            purchase_date TEXT NOT NULL,
            litres REAL NOT NULL,
            cost REAL NOT NULL,
            odometer REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY,
            monthly_budget REAL NOT NULL
        )
    """)

    conn.commit()

    return conn


conn = get_connection()


# ============================================================
# 3. VEHICLE MANAGEMENT
# ============================================================

def save_vehicle(model, efficiency, tank_capacity):

    conn.execute("""
        INSERT INTO vehicles
        (id, model, efficiency, tank_capacity)

        VALUES (1, ?, ?, ?)

        ON CONFLICT(id)
        DO UPDATE SET
            model = excluded.model,
            efficiency = excluded.efficiency,
            tank_capacity = excluded.tank_capacity
    """, (model, efficiency, tank_capacity))

    conn.commit()


def get_vehicle():

    return conn.execute("""
        SELECT model, efficiency, tank_capacity
        FROM vehicles
        WHERE id = 1
    """).fetchone()


# ============================================================
# 4. FUEL RECORD MANAGEMENT
# ============================================================

def add_fuel_record(purchase_date, litres, cost, odometer):

    conn.execute("""
        INSERT INTO fuel_records
        (purchase_date, litres, cost, odometer)

        VALUES (?, ?, ?, ?)
    """, (
        str(purchase_date),
        litres,
        cost,
        odometer
    ))

    conn.commit()


def get_fuel_records():

    return pd.read_sql_query("""
        SELECT *
        FROM fuel_records
        ORDER BY purchase_date DESC, id DESC
    """, conn)


# ============================================================
# 5. MONTHLY FUEL BUDGET
# ============================================================

def save_budget(amount):

    conn.execute("""
        INSERT INTO budgets (id, monthly_budget)

        VALUES (1, ?)

        ON CONFLICT(id)
        DO UPDATE SET
            monthly_budget = excluded.monthly_budget
    """, (amount,))

    conn.commit()


def get_budget():

    result = conn.execute("""
        SELECT monthly_budget
        FROM budgets
        WHERE id = 1
    """).fetchone()

    return float(result[0]) if result else 0.0


def get_monthly_spending():

    current_month = date.today().strftime("%Y-%m")

    result = conn.execute("""
        SELECT COALESCE(SUM(cost), 0)

        FROM fuel_records

        WHERE substr(purchase_date, 1, 7) = ?
    """, (current_month,)).fetchone()

    return float(result[0])


# ============================================================
# 6. GENERATE EDUCATIONAL MACHINE LEARNING DATASET
# ============================================================

def generate_dataset():

    if DATA_PATH.exists():
        return

    random.seed(42)

    rows = []

    for i in range(500):

        efficiency = random.uniform(7, 22)

        speed = random.uniform(15, 110)

        load = random.uniform(0, 500)

        ac = random.randint(0, 1)

        road = random.choice([0, 1, 2])

        # Base fuel consumption
        consumption = 100 / efficiency

        # Operating-condition adjustments
        speed_factor = 0.000108 * (speed - 70) ** 2

        road_factor = {
            0: 1.20,
            1: 1.00,
            2: 0.93
        }[road]

        load_factor = 1 + (0.00018 * load)

        ac_factor = 1.07 if ac else 1.0

        final_consumption = (
            consumption + speed_factor
        ) * road_factor * load_factor * ac_factor

        final_consumption += random.gauss(0, 0.20)

        final_consumption = max(2, final_consumption)

        rows.append([
            efficiency,
            speed,
            load,
            ac,
            road,
            final_consumption
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "efficiency",
            "speed",
            "load",
            "ac",
            "road",
            "fuel_consumption"
        ]
    )

    df.to_csv(DATA_PATH, index=False)


generate_dataset()


# ============================================================
# 7. ARTIFICIAL INTELLIGENCE: MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = pd.read_csv(DATA_PATH)

    X = data[
        [
            "efficiency",
            "speed",
            "load",
            "ac",
            "road"
        ]
    ]

    y = data["fuel_consumption"]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model()


# ============================================================
# 8. FUEL CONSUMPTION PREDICTION
# ============================================================

ROAD_TYPES = {
    "City": 0,
    "Mixed": 1,
    "Highway": 2
}


def predict_fuel(
    distance,
    efficiency,
    speed,
    load,
    ac,
    road
):

    input_data = pd.DataFrame(
        [[
            efficiency,
            speed,
            load,
            int(ac),
            ROAD_TYPES[road]
        ]],

        columns=[
            "efficiency",
            "speed",
            "load",
            "ac",
            "road"
        ]
    )

    predicted_consumption = float(
        model.predict(input_data)[0]
    )

    predicted_consumption = max(
        0.1,
        predicted_consumption
    )

    required_fuel = (
        distance * predicted_consumption / 100
    )

    return {
        "distance_km": distance,
        "fuel_litres": required_fuel,
        "consumption_l_100km": predicted_consumption
    }


# ============================================================
# 9. FUEL PRICE CALCULATION
# ============================================================

def get_average_fuel_price():

    records = get_fuel_records()

    if records.empty:

        # Illustrative default price in PKR per litre
        return 280.0

    total_litres = records["litres"].sum()

    total_cost = records["cost"].sum()

    if total_litres <= 0:
        return 280.0

    return total_cost / total_litres


# ============================================================
# 10. AGENTIC AI: PYTHON TOOLS
# ============================================================

# TOOL 1: Retrieve vehicle information

def tool_get_vehicle_profile():

    vehicle = get_vehicle()

    if not vehicle:

        return {
            "error": "No vehicle profile found. Save a vehicle first."
        }

    return {
        "vehicle_model": vehicle[0],
        "fuel_efficiency_km_per_litre": vehicle[1],
        "tank_capacity_litres": vehicle[2]
    }


# TOOL 2: Estimate fuel consumption

def tool_estimate_fuel(
    distance_km,
    average_speed_kmph,
    additional_load_kg,
    air_conditioning,
    road_type
):

    vehicle = get_vehicle()

    if not vehicle:

        return {
            "error": "Vehicle profile is missing."
        }

    prediction = predict_fuel(
        distance=distance_km,
        efficiency=vehicle[1],
        speed=average_speed_kmph,
        load=additional_load_kg,
        ac=air_conditioning,
        road=road_type
    )

    price = get_average_fuel_price()

    prediction["estimated_cost_pkr"] = (
        prediction["fuel_litres"] * price
    )

    return prediction


# TOOL 3: Check fuel budget

def tool_check_budget(projected_cost_pkr):

    budget = get_budget()

    spending = get_monthly_spending()

    projected_total = spending + projected_cost_pkr

    return {
        "monthly_budget_pkr": budget,
        "current_month_spending_pkr": spending,
        "projected_trip_cost_pkr": projected_cost_pkr,
        "projected_total_spending_pkr": projected_total,
        "budget_exceeded": (
            budget > 0 and projected_total > budget
        )
    }


# ============================================================
# 11. OPENAI AGENT FUNCTION DEFINITIONS
# ============================================================

AGENT_TOOLS = [

    {
        "type": "function",

        "name": "get_vehicle_profile",

        "description": (
            "Retrieve the saved vehicle model, "
            "fuel efficiency and tank capacity."
        ),

        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },

        "strict": True
    },

    {
        "type": "function",

        "name": "estimate_fuel",

        "description": (
            "Predict fuel consumption for a trip "
            "using the saved vehicle and ML model."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "distance_km": {
                    "type": "number",
                    "description": "Trip distance in kilometres."
                },

                "average_speed_kmph": {
                    "type": "number",
                    "description": "Average speed in km/h."
                },

                "additional_load_kg": {
                    "type": "number",
                    "description": "Additional vehicle load in kg."
                },

                "air_conditioning": {
                    "type": "boolean",
                    "description": "Whether AC is used."
                },

                "road_type": {
                    "type": "string",
                    "enum": ["City", "Mixed", "Highway"]
                }

            },

            "required": [
                "distance_km",
                "average_speed_kmph",
                "additional_load_kg",
                "air_conditioning",
                "road_type"
            ],

            "additionalProperties": False
        },

        "strict": True
    },

    {
        "type": "function",

        "name": "check_fuel_budget",

        "description": (
            "Check whether the predicted trip cost "
            "will exceed the monthly fuel budget."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "projected_cost_pkr": {
                    "type": "number",
                    "description": "Estimated trip cost in PKR."
                }

            },

            "required": ["projected_cost_pkr"],

            "additionalProperties": False
        },

        "strict": True
    }

]


# ============================================================
# 12. AGENT TOOL EXECUTION CONTROLLER
# ============================================================

def execute_agent_tool(name, arguments):

    if name == "get_vehicle_profile":

        return tool_get_vehicle_profile()

    elif name == "estimate_fuel":

        return tool_estimate_fuel(
            **arguments
        )

    elif name == "check_fuel_budget":

        return tool_check_budget(
            **arguments
        )

    return {
        "error": "Unknown tool requested."
    }


# ============================================================
# 13. OPENAI AGENT: PLAN, ACT, OBSERVE, RESPOND
# ============================================================

def run_openai_agent(user_message, chat_history):

    if not OPENAI_API_KEY:

        raise ValueError(
            "OpenAI API key is missing. "
            "Add OPENAI_API_KEY to your .env file."
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=60.0,
        max_retries=2
    )

    instructions = """
    You are Smart Fuel AI, an intelligent automotive
    fuel-efficiency and resource-management assistant.

    You have expertise in:
    - Mechanical Engineering
    - Vehicle fuel consumption
    - Automotive energy efficiency
    - Fuel budgeting
    - Artificial Intelligence
    - Machine Learning

    You also operate as an Agentic AI system.

    IMPORTANT AGENT INSTRUCTIONS:

    1. Understand the user's objective.
    2. When a trip estimate is requested, retrieve
       the vehicle profile using get_vehicle_profile.
    3. Use estimate_fuel to obtain an ML-based prediction.
    4. Use check_fuel_budget to evaluate projected spending.
    5. Analyze the results returned by the tools.
    6. Provide a clear and technically meaningful recommendation.

    If trip conditions are missing, use these assumptions:
    - Average speed: 60 km/h
    - Additional load: 100 kg
    - Air conditioning: ON
    - Road type: Mixed

    Clearly disclose any assumptions.

    Never invent tool results.
    Never claim to have booked a route or made a payment.
    Never modify or delete user records.

    Explain engineering concepts in simple language.
    Always distinguish predicted results from actual measurements.
    """

    # Include recent conversation context
    messages = chat_history[-8:] + [
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = client.responses.create(

        model=OPENAI_MODEL,

        instructions=instructions,

        input=messages,

        tools=AGENT_TOOLS,

        tool_choice="auto",

        max_output_tokens=1200
    )

    execution_log = []

    # Agent execution loop
    for step in range(5):

        function_calls = [

            item for item in response.output

            if item.type == "function_call"

        ]

        if not function_calls:

            return (
                response.output_text,
                execution_log
            )

        tool_outputs = []

        for call in function_calls:

            try:

                arguments = json.loads(
                    call.arguments
                )

                result = execute_agent_tool(
                    call.name,
                    arguments
                )

            except Exception as error:

                result = {
                    "error": str(error)
                }

            execution_log.append({
                "tool": call.name,
                "arguments": arguments if "arguments" in locals() else {},
                "result": result
            })

            tool_outputs.append({

                "type": "function_call_output",

                "call_id": call.call_id,

                "output": json.dumps(
                    result,
                    ensure_ascii=False
                )

            })

        # Send tool results back to the model
        response = client.responses.create(

            model=OPENAI_MODEL,

            instructions=instructions,

            previous_response_id=response.id,

            input=tool_outputs,

            tools=AGENT_TOOLS,

            tool_choice="auto",

            max_output_tokens=1200
        )

    raise RuntimeError(
        "Agent reached its maximum tool-execution steps."
    )


# ============================================================
# 14. SIDEBAR: VEHICLE AND BUDGET SETTINGS
# ============================================================

st.sidebar.title("⚙️ System Settings")

st.sidebar.subheader("Vehicle Information")

existing_vehicle = get_vehicle()

if existing_vehicle:

    default_model = existing_vehicle[0]
    default_efficiency = float(existing_vehicle[1])
    default_tank = float(existing_vehicle[2])

else:

    default_model = "Toyota Corolla"
    default_efficiency = 12.0
    default_tank = 50.0


with st.sidebar.form("vehicle_form"):

    vehicle_model = st.text_input(
        "Vehicle Model",
        value=default_model
    )

    vehicle_efficiency = st.number_input(
        "Fuel Efficiency (km/L)",
        min_value=1.0,
        max_value=60.0,
        value=default_efficiency
    )

    tank_capacity = st.number_input(
        "Tank Capacity (Litres)",
        min_value=5.0,
        max_value=300.0,
        value=default_tank
    )

    save_vehicle_button = st.form_submit_button(
        "Save Vehicle"
    )


if save_vehicle_button:

    if vehicle_model.strip():

        save_vehicle(
            vehicle_model.strip(),
            vehicle_efficiency,
            tank_capacity
        )

        st.sidebar.success(
            "Vehicle information saved."
        )

    else:

        st.sidebar.error(
            "Please enter a vehicle model."
        )


st.sidebar.divider()

st.sidebar.subheader("Monthly Fuel Budget")

current_budget = get_budget()

with st.sidebar.form("budget_form"):

    budget_amount = st.number_input(
        "Budget (PKR)",
        min_value=0.0,
        value=current_budget,
        step=1000.0
    )

    save_budget_button = st.form_submit_button(
        "Save Budget"
    )


if save_budget_button:

    save_budget(budget_amount)

    st.sidebar.success(
        "Monthly budget updated."
    )


st.sidebar.divider()

if OPENAI_API_KEY:

    st.sidebar.success(
        "OpenAI API key detected."
    )

else:

    st.sidebar.error(
        "OpenAI API key not found."
    )

st.sidebar.caption(
    f"AI Model: {OPENAI_MODEL}"
)


# ============================================================
# 15. MAIN DASHBOARD
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([

    "📊 Dashboard",

    "🚗 Trip Planner",

    "⛽ Fuel Records",

    "🤖 Agentic AI Assistant"

])


records = get_fuel_records()

monthly_spending = get_monthly_spending()

monthly_budget = get_budget()

fuel_price = get_average_fuel_price()


# ---------------- DASHBOARD ----------------

with tab1:

    st.subheader(
        "Fuel Performance Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    total_litres = (
        records["litres"].sum()
        if not records.empty else 0
    )

    total_cost = (
        records["cost"].sum()
        if not records.empty else 0
    )

    col1.metric(
        "Total Fuel Purchased",
        f"{total_litres:.2f} L"
    )

    col2.metric(
        "Total Fuel Expenses",
        f"PKR {total_cost:,.0f}"
    )

    col3.metric(
        "Current Month Spending",
        f"PKR {monthly_spending:,.0f}"
    )

    col4.metric(
        "Monthly Fuel Budget",
        f"PKR {monthly_budget:,.0f}"
    )

    st.divider()

    if monthly_budget > 0:

        utilization = (
            monthly_spending / monthly_budget
        )

        st.write("Monthly Budget Utilization")

        st.progress(
            min(utilization, 1.0)
        )

        st.write(
            f"{utilization * 100:.1f}% of the monthly budget used."
        )

        if utilization > 1:

            st.error(
                "Monthly fuel budget exceeded!"
            )

        elif utilization >= 0.8:

            st.warning(
                "Fuel expenditure is approaching the budget limit."
            )

        else:

            st.success(
                "Fuel expenditure is within the budget."
            )

    if not records.empty:

        chart_data = records.copy()

        chart_data["purchase_date"] = pd.to_datetime(
            chart_data["purchase_date"]
        )

        daily_data = (
            chart_data.groupby("purchase_date")["litres"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots()

        ax.plot(
            daily_data["purchase_date"],
            daily_data["litres"],
            marker="o"
        )

        ax.set_title(
            "Fuel Purchase Trend"
        )

        ax.set_xlabel(
            "Purchase Date"
        )

        ax.set_ylabel(
            "Fuel Purchased (Litres)"
        )

        ax.grid(True)

        plt.xticks(rotation=30)

        st.pyplot(fig)

        st.subheader(
            "Recent Fuel Records"
        )

        st.dataframe(
            records,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Add fuel purchase records to display analytics."
        )


# ---------------- TRIP PLANNER ----------------

with tab2:

    st.subheader(
        "AI-Based Fuel Consumption Estimator"
    )

    vehicle = get_vehicle()

    if not vehicle:

        st.warning(
            "Save your vehicle information in the sidebar first."
        )

    else:

        st.write(
            f"Vehicle: **{vehicle[0]}**"
        )

        st.write(
            f"Baseline efficiency: **{vehicle[1]} km/L**"
        )

        with st.form("trip_form"):

            distance = st.number_input(
                "Trip Distance (km)",
                min_value=1.0,
                max_value=3000.0,
                value=100.0
            )

            speed = st.slider(
                "Average Speed (km/h)",
                10,
                120,
                60
            )

            load = st.number_input(
                "Additional Load (kg)",
                min_value=0.0,
                max_value=2000.0,
                value=100.0
            )

            ac = st.checkbox(
                "Air Conditioning ON",
                value=True
            )

            road = st.selectbox(
                "Road Type",
                ["City", "Mixed", "Highway"]
            )

            predict_button = st.form_submit_button(
                "Predict Fuel Consumption"
            )

        if predict_button:

            result = predict_fuel(
                distance,
                vehicle[1],
                speed,
                load,
                ac,
                road
            )

            estimated_cost = (
                result["fuel_litres"] * fuel_price
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Estimated Fuel",
                f"{result['fuel_litres']:.2f} L"
            )

            c2.metric(
                "Fuel Consumption",
                f"{result['consumption_l_100km']:.2f} L/100 km"
            )

            c3.metric(
                "Estimated Trip Cost",
                f"PKR {estimated_cost:,.0f}"
            )

            st.success(
                "Machine-learning prediction completed."
            )

            st.caption(
                "Prediction is an educational estimate, "
                "not a measured fuel-consumption result."
            )


# ---------------- FUEL RECORDS ----------------

with tab3:

    st.subheader(
        "Fuel Purchase Management"
    )

    with st.form("fuel_record_form"):

        purchase_date = st.date_input(
            "Purchase Date",
            value=date.today()
        )

        litres = st.number_input(
            "Fuel Quantity (Litres)",
            min_value=0.1,
            max_value=500.0,
            value=20.0
        )

        cost = st.number_input(
            "Total Cost (PKR)",
            min_value=1.0,
            max_value=1000000.0,
            value=5600.0
        )

        odometer = st.number_input(
            "Odometer Reading (km, 0 if unknown)",
            min_value=0.0,
            value=0.0
        )

        save_record = st.form_submit_button(
            "Save Fuel Record"
        )

    if save_record:

        add_fuel_record(
            purchase_date,
            litres,
            cost,
            odometer if odometer > 0 else None
        )

        st.success(
            "Fuel record saved successfully."
        )

        st.rerun()

    st.info(
        f"Average recorded fuel price: PKR {fuel_price:.2f}/L. "
        "If no records exist, an illustrative default price is used."
    )


# ---------------- AGENTIC AI ASSISTANT ----------------

with tab4:

    st.subheader(
        "🤖 Smart Fuel Agent"
    )

    st.markdown(
        """
        This assistant uses OpenAI function calling to interact
        with the project's Python tools.

        It can retrieve vehicle information, estimate fuel
        consumption and evaluate monthly fuel expenditure.
        """
    )

    if "agent_history" not in st.session_state:

        st.session_state.agent_history = []

    # Display previous conversation
    for message in st.session_state.agent_history:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

    with st.form("agent_chat_form", clear_on_submit=True):

        user_question = st.text_area(
            "Ask Smart Fuel AI",
            placeholder=(
                "Example: Estimate the fuel required "
                "for a 150 km trip and check my budget."
            ),
            height=100
        )

        send_button = st.form_submit_button(
            "Run AI Agent"
        )

    if send_button and user_question.strip():

        st.session_state.agent_history.append({

            "role": "user",

            "content": user_question

        })

        with st.chat_message("user"):

            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner(
                "AI Agent is planning and executing tools..."
            ):

                try:

                    answer, execution_log = run_openai_agent(

                        user_question,

                        st.session_state.agent_history[:-1]

                    )

                    st.markdown(answer)

                    st.session_state.agent_history.append({

                        "role": "assistant",

                        "content": answer

                    })

                    if execution_log:

                        with st.expander(
                            "Agent Execution Trace",
                            expanded=True
                        ):

                            for index, entry in enumerate(
                                execution_log,
                                start=1
                            ):

                                st.markdown(
                                    f"**Step {index}: "
                                    f"{entry['tool']}**"
                                )

                                st.json(entry)

                            st.success(
                                "Agent tool execution completed."
                            )

                    else:

                        st.caption(
                            "The assistant answered without calling a tool."
                        )

                except Exception as error:

                    st.error(
                        f"Agent Error: {error}"
                    )

    if st.button("Clear AI Conversation"):

        st.session_state.agent_history = []

        st.rerun()


# ============================================================
# 16. PROJECT INFORMATION
# ============================================================

with st.expander(
    "Project Methodology and Technical Information"
):

    st.markdown(
        """
        ### Artificial Intelligence

        A Random Forest Regression model predicts fuel
        consumption using:

        - Vehicle fuel efficiency
        - Average vehicle speed
        - Additional vehicle load
        - Air-conditioning status
        - Road type

        ### Agentic Artificial Intelligence

        The OpenAI-powered agent follows:

        1. **Planning:** Understands the user's request.
        2. **Tool selection:** Selects appropriate Python functions.
        3. **Execution:** Retrieves vehicle data and predicts fuel use.
        4. **Observation:** Checks the returned results and budget.
        5. **Recommendation:** Generates an intelligent response.

        ### Database

        SQLite stores:

        - Vehicle information
        - Fuel purchase records
        - Monthly fuel budgets

        ### Limitations

        - The ML training data is synthetic and educational.
        - Fuel consumption predictions require validation
          against actual vehicle data.
        - Live GPS and traffic-based route optimization
          are not implemented.
        - The agent cannot make payments or modify records.
        - All recommendations require user judgment.
        """
    )

st.divider()

st.caption(
    "Smart Fuel AI | Huawei AI Course Project | "
    "Mechanical Engineering Application"
)
