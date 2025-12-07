import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(layout= 'wide', page_title='Suicides Rate  EDA')
html_title = """<h1 style="color:white;text-align:center;"><span style="color:green">🚲 Bike Rental </span>Exploratory Data Analysis (EDA) </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('https://th.bing.com/th/id/R.bca760e7e92ee225feb8ccabf69b3cd5?rik=lJGhRwaGTBrDSg&pid=ImgRaw&r=0')


df = pd.read_csv("Cleaned_Bike.csv",index_col=0)
page = st.sidebar.radio('Pages', ['Home', 'Analysis charts', "Preprocessing --> ML"])
if page=='Home':
    st.title("📊 Dataset View & Advanced Filtering")
    st.subheader('Dataset Overview')
    st.dataframe(df)
    st.subheader('display the number of the Rows and Columns')
    st.write(f'Number Of Rows : {df.shape[0]}')
    st.write(f'Number Of columns :{df.shape[1]}')
    st.write('---')

    columns_info = {
        "datetime": ("Date and time of data record", "2011-01-01 00:00:00"),
        "count": ("Total number of rented bikes in the current hour", "254"),
        "holiday": ("Indicates whether the day is a public holiday (0: No, 1: Yes)", "0"),
        "workingday": ("Indicates if the day is a working day (1: Yes, 0: Weekend/Holiday)", "1"),
        "temp": ("Temperature value in Celsius or normalized", "15.3"),
        "feels_like": ("Human perception of temperature (feels-like)", "16.0"),
        "temp_min": ("Minimum temperature recorded during the day", "13.5"),
        "temp_max": ("Maximum temperature recorded during the day", "18.2"),
        "pressure": ("Atmospheric pressure (hPa)", "1015"),
        "humidity": ("Percentage of humidity in the air", "65"),
        "wind_speed": ("Wind speed value", "4.5"),
        "wind_deg": ("Wind direction in degrees", "230"),
        "clouds_all": ("Cloudiness percentage of the sky", "75"),
        "weather_main": ("Main weather condition", "Clear"),
        "hour": ("Hour of the day (0–23)", "14"),
        "day": ("Day of the month (1–31)", "10"),
        "month": ("Month number (1–12)", "7"),
        "year": ("Year of record", "2012"),
        "weather_temp": ("Temperature from weather API source", "14.8"),
        "Today_times": ("Number of recorded hours in the same day", "9"),
        "season": ("Season of the year (Spring, Summer, Fall, Winter)", "Spring"),
        "heat_feeling": ("Subjective heat comfort category", "Warm")
    }

    selected_col = st.sidebar.selectbox("Select Column", df.columns)
    st.title(f"📌 {selected_col}")

    description = columns_info.get(selected_col, "No description available")
    st.write(f"Definition: {description}")

    # Show example value
    example_value = df[selected_col].dropna().iloc[0] if df[selected_col].notna().any() else "No example"
    st.write(f"Example: {example_value}")


    if selected_col == 'datetime':
        df['datetime_str'] = pd.to_datetime(df['datetime']).astype(str)
        plot_col = 'datetime_str'
    else:
        plot_col = selected_col

    if pd.api.types.is_numeric_dtype(df[plot_col]):
        st.subheader("📊 Histogram")
        fig = px.histogram(
            df[plot_col].dropna(),
            x=plot_col,
            nbins=20,
            title=f"{plot_col} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📊 Count Plot")

        df_count = df[plot_col].dropna().value_counts().reset_index()
        df_count.columns = [plot_col, 'value_count']

        fig = px.bar(
            df_count,
            x=plot_col,
            y='value_count',
            title=f"{plot_col} Counts"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])


    if "date" in df.columns:
        st.sidebar.subheader("📅 Date Range")
        min_date = df["date"].min()
        max_date = df["date"].max()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        df_filtered = df[
            (df["date"] >= pd.to_datetime(date_range[0])) &
            (df["date"] <= pd.to_datetime(date_range[1]))
        ]
    else:
        df_filtered = df.copy()

    if "weather_main" in df.columns:
        st.sidebar.subheader("⛅ Weather Condition")
        weather_opt = st.sidebar.multiselect(
            "Weather",
            options=df["weather_main"].unique(),
            default=df["weather_main"].unique()
        )
        df_filtered = df_filtered[df_filtered["weather_main"].isin(weather_opt)]

    if "holiday" in df.columns:
        st.sidebar.subheader("🎉 Holiday Filter")
        holiday_opt = st.sidebar.radio(
            "Holiday?",
            options=["All", "Holiday Only", "Non-Holiday"]
        )
        if holiday_opt == "Holiday Only":
            df_filtered = df_filtered[df_filtered["holiday"] == 1]
        elif holiday_opt == "Non-Holiday":
            df_filtered = df_filtered[df_filtered["holiday"] == 0]

    if "hour" in df.columns:
        st.sidebar.subheader("⏱ Hour Filter")
        hour_range = st.sidebar.slider(
            "Hour Range",
            min_value=int(df["hour"].min()),
            max_value=int(df["hour"].max()),
            value=(0, 23)
        )
        df_filtered = df_filtered[
            (df_filtered["hour"] >= hour_range[0]) &
            (df_filtered["hour"] <= hour_range[1])
        ]


    if "count" in df.columns:
        st.sidebar.subheader("🚲 Count Filter")
        count_range = st.sidebar.slider(
            "Bike Count Range",
            min_value=int(df["count"].min()),
            max_value=int(df["count"].max()),
            value=(int(df["count"].min()), int(df["count"].max()))
        )
        df_filtered = df_filtered[
            (df_filtered["count"] >= count_range[0]) &
            (df_filtered["count"] <= count_range[1])
        ]


    st.write("### 📋 Sample Data")
    st.dataframe(df_filtered.head(10))

    st.write("### 🧭 Summary Statistics")
    st.write(df_filtered.describe())    
elif page =='Analysis charts':

    @st.cache_data
    def load_data():
        return pd.read_csv("Cleaned_Bike.csv")

    df = load_data()


    st.sidebar.header("Filters")


    hour_min, hour_max = int(df["hour"].min()), int(df["hour"].max())
    hour_filter = st.sidebar.slider("Select Hour Range", hour_min, hour_max, (hour_min, hour_max))


    holiday_filter = st.sidebar.multiselect(
        "Holiday",
        options=df["holiday"].unique(),
        default=df["holiday"].unique()
    )


    workingday_filter = st.sidebar.multiselect(
        "Working Day",
        options=df["workingday"].unique(),
        default=df["workingday"].unique()
    )


    weather_filter = st.sidebar.multiselect(
        "Weather Main",
        options=df["weather_main"].unique(),
        default=df["weather_main"].unique()
    )


    season_filter = st.sidebar.multiselect(
        "Season",
        options=df["season"].unique(),
        default=df["season"].unique()
    )


    humidity_min, humidity_max = df["humidity"].min(), df["humidity"].max()
    humidity_filter = st.sidebar.slider("Humidity Range", float(humidity_min), float(humidity_max), (float(humidity_min), float(humidity_max)))


    filtered_df = df[
        (df["hour"].between(hour_filter[0], hour_filter[1])) &
        (df["holiday"].isin(holiday_filter)) &
        (df["workingday"].isin(workingday_filter)) &
        (df["weather_main"].isin(weather_filter)) &
        (df["season"].isin(season_filter)) &
        (df["humidity"].between(humidity_filter[0], humidity_filter[1]))
    ]

    st.title("📊 Bike Sharing Data Analysis")


    st.subheader("⏰ Bike Rental per Hour")
    hour_co = filtered_df.groupby("hour")["count"].sum().reset_index()
    fig = px.line(hour_co, x='hour', y='count')
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("🏖️ Bike Rental on Holidays")
    holiday = filtered_df.groupby("holiday")["count"].mean().reset_index()
    fig = px.bar(holiday, x='holiday', y='count', text_auto=True,color_discrete_sequence=['red'])
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("💼 Bike Rental on Working Days")
    working = filtered_df.groupby("workingday")["count"].mean().reset_index()
    fig = px.bar(working, x='workingday', y='count', text_auto=True,color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("📅 Working Day + Holiday Effect")
    holi_work = filtered_df.groupby(["holiday","workingday"])["count"].mean().reset_index()
    fig = px.bar(holi_work, x='holiday', y='count', color='workingday', barmode='group')
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("🌦️ Weather Effect on Bike Rental")
    weather = filtered_df.groupby(["weather_temp","weather_main"])["count"].mean().reset_index()
    fig = px.bar(weather, x='weather_temp', y='count', color='weather_main')
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("🍂 Bike Rental per Season")
    season = filtered_df.groupby("season")["count"].sum().reset_index()
    fig = px.funnel(season, x='season', y='count',color_discrete_sequence=['blue'])
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("💧 Bike Rental vs Humidity")
    humidity = filtered_df.groupby("humidity")["count"].mean().reset_index()
    fig = px.line(humidity, x='humidity', y='count')
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("☁️ Effect of Cloud Coverage")
    clouds = filtered_df.groupby("clouds_all")["count"].mean().reset_index()
    fig = px.line(clouds, x="clouds_all", y="count")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📈 Bike Rental Per Days and Hours")
    top_days=filtered_df.groupby(['hour','day'])['count'].sum().reset_index().sort_values(by='count',ascending=False)
    fig=px.bar(data_frame=top_days,x='hour',y='count',color='day')
    st.plotly_chart(fig)

    st.subheader("📈 Correlation Matrix")
    corr = filtered_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True,width=1100,height=900)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Average Rentals by Weather Condition and Temperature Level")
    pivot_table=df.pivot_table(values='count',index='weather_main',columns='weather_temp',aggfunc='mean')
    fig=px.imshow(pivot_table,text_auto=True,width=1050,height=700)
    st.plotly_chart(fig)
elif page=='Preprocessing --> ML':

    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

    st.title("Data Preprocessing Dashboard")

    st.subheader("Original Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.subheader("Feature Scaling (Numerical Columns)")
        st.write("Numeric columns:", numeric_cols)
        sc = StandardScaler()
        df[numeric_cols] = sc.fit_transform(df[numeric_cols])
        st.dataframe(df[numeric_cols].head())

    ordinal_cols = ['season', 'Today_times']
    season_ca = ['Spring', 'Summer', 'Autumn', 'Winter']
    today_times_ca = ['Morning', 'Afternoon', 'Evening', 'Night']
    categories_1 = [season_ca, today_times_ca]

    ordinal_cols_exist = [col for col in ordinal_cols if col in df.columns]
    if ordinal_cols_exist:
        st.subheader("Ordinal Encoding")
        st.write("Ordinal columns:", ordinal_cols_exist)
        ordinal_encoder = OrdinalEncoder(categories=categories_1, handle_unknown='use_encoded_value', unknown_value=-1)
        df[ordinal_cols_exist] = ordinal_encoder.fit_transform(df[ordinal_cols_exist])
        st.dataframe(df[ordinal_cols_exist].head())

    nominal_cols = ['holiday', 'workingday', 'weather_main']
    nominal_cols_exist = [col for col in nominal_cols if col in df.columns]
    if nominal_cols_exist:
        st.subheader("One-Hot Encoding")
        st.write("Nominal columns:", nominal_cols_exist)
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded = ohe.fit_transform(df[nominal_cols_exist])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(nominal_cols_exist))
        df = df.drop(nominal_cols_exist, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        st.dataframe(encoded_df.head())

    st.subheader("Final Preprocessed Data")
    st.dataframe(df.head())

