import streamlit as st
import pandas as pd
import pickle
import datetime
import joblib
from keras.models import load_model
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from tools import *

import matplotlib.pyplot as plt
import seaborn as sns
# import pywhatkit
# import win32clipboard
# import pyautogui
# import keyboard

# page config
page_config(title="Flood - QuakeFlood Alert")
sns.set_style("darkgrid")

# function tab 1
def plot_tab1(range, data_plot):
    st.caption(f"Here's the height chart for {range}")

    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(data_plot['height (cm)'], color='cornflowerblue', alpha=1, linewidth=3, label='height')
    plt.fill_between(x=data_plot.index,y1=data_plot['height (cm)'], color='cornflowerblue', alpha=0.4)

    plt.xlim(data_plot.index.min(), data_plot.index.max())
    plt.ylim(0, data_plot['height (cm)'].max()+20)
    plt.ylabel('Height (cm)')
    plt.xlabel('Datetime')
    sns.despine(left=True, right=True)
    st.pyplot(fig1)
def convert_df(df):
        return df.to_csv().encode('utf-8')
def download_csv(data_df, name):
    st.dataframe(data_df)
    st.caption("Need the data?")
    if st.download_button(label="Download data as CSV",
                           data=convert_df(data_df),
                           file_name=name,
                           mime='text/csv'):
        st.toast('Download complete!', icon = "✅")

# function tab 2
def data_plot_tab2(history, pred):
    # history
    data_history = history.reset_index()
    data_history['index'] = data_history['index'] - 287
    data_history = data_history.set_index('index')
    data_plot_X = data_history[['height (cm)']]
    data_plot_X = data_plot_X.rename(columns = {'height (cm)':'height_history (cm)'})
    # future
    data_pred = pred.reset_index()
    data_pred['index'] = data_pred['index'] + 1
    data_pred = data_pred.set_index('index')
    #concat
    data_plot_2 = pd.concat([data_plot_X, data_pred])
    return data_history, data_pred, data_plot_2

# function cache
@st.cache_resource(ttl=3600)
def load_data_banjir():
     with open("dataset/data_simulasi_banjir.pkl", 'rb') as file:
         data = pickle.load(file)
     return data

@st.cache_resource(ttl=3600)
def load_scaler_banjir():
     scaler_X_klas = joblib.load('scaler/scaler_X_klasifikasi_banjir.save')
     scaler_X_pred = joblib.load('scaler/scaler_X_prediksi_banjir.save') 
     scaler_y_pred = joblib.load('scaler/scaler_y_prediksi_banjir.save') 
     return scaler_X_klas, scaler_X_pred, scaler_y_pred

@st.cache_resource(ttl=3600)
def load_model_banjir():
     model_klas = load_model('model/model_klasifikasi_banjir.h5')
     model_pred = load_model('model/model_prediksi_banjir.h5')
     return model_klas, model_pred


# load data, scaler, model
data_banjir = load_data_banjir()
scaler_X_klasifikasi, scaler_X_prediksi, scaler_y_prediksi = load_scaler_banjir()
model_klasifikasi_banjir, model_prediksi_banjir = load_model_banjir()

# title
st.title("Flood")
# tabs
tab1, tab2, tab3= st.tabs(["**Dashboard**", "**Prediction**", "**Send Message**"])

# tab 1 (Dashboard)
with tab1:
    col_datetime1, col_metric1, col_metric2= st.columns(3)
    # datetime
    with col_datetime1:
        date1 = st.date_input("d", 
                               value=datetime.datetime(2022,9,30),
                               min_value=datetime.datetime(2022,9,9),
                               max_value=datetime.datetime(2022,9,30),
                               label_visibility="collapsed")
        time1 = st.time_input("t",
                               value=datetime.time(10,30),
                               step=600,
                               label_visibility="collapsed")
        st.caption("Set date and time 👆")
    # data tab 1
    datetime1 = str(date1)+" "+str(time1)
    data_tab1 = data_banjir.loc[data_banjir['datetime']<=datetime1].reset_index(drop=True)
    data_metric = data_tab1[['datetime','height (cm)']].head(2)
    data_metric = data_metric.rename(columns={'datetime':'date',
                                             'height (cm)':'height'})
    data_metric = klasifikasi_banjir(X=data_metric,        # get status
                                     scaler_X=scaler_X_klasifikasi,
                                     model=model_klasifikasi_banjir)
    
    metric1_value = data_metric.height[0].round(2)
    metric1_delta = (data_metric.height[0] - data_metric.height[1]).round(2)
    
    metric2_value = data_metric.status_pred[0]
    metric2_delta = int(metric2_value - data_metric.status_pred[1])

    # metric 1
    with col_metric1:
        st.metric(label="Last height:",
                  value=f"{metric1_value} cm",
                  delta=metric1_delta,
                  delta_color="inverse")
    # metric 2
    with col_metric2:
        st.metric(label="Last status:",
                  value=f"Siaga {metric2_value}",
                  delta=metric2_delta,
                  delta_color="off")
    # metric style
    style_metric_cards(background_color="#f0f2f6",
                       border_left_color="#E84545",
                       border_size_px=0, 
                       border_radius_px=10,
                       box_shadow=False)

    # plot
    data_plot1 = data_tab1[['datetime','height (cm)']].set_index("datetime")        # set 'date' as index
    col_plot1,col_plot2 = st.columns([2,1])
    col_plot1.subheader("Visualizing height changes:")
    data_plot1_range = col_plot2.selectbox("Plot range:", 
                        ("last day","last week","last month","last year","all data"),
                        label_visibility="collapsed")
    if data_plot1_range == "last day":                     # last day
        plot_tab1(range="the last day",
                  data_plot=data_plot1.head(144))
    elif data_plot1_range == "last week":                  # last week
        plot_tab1(range="the last week",
                  data_plot=data_plot1.head(1008))
    elif data_plot1_range == "last month":                 # last month
        plot_tab1(range="the last month",
                  data_plot=data_plot1.head(4320))
    elif data_plot1_range == "last year":                  # last year
        plot_tab1(range="the last year",
                  data_plot=data_plot1.head(52560))     
    elif data_plot1_range == "all data":                   # all data
        plot_tab1(range="all data",
                  data_plot=data_plot1)

    # data history
    col_data1,col_data2 = st.columns([2,1])
    col_data1.subheader("Historical data table:")
    data_tab1_range = col_data2.selectbox("Data range:", 
                        ("last day","last week","last month","last year","all data"),
                        label_visibility="collapsed")
    if data_tab1_range == "last day":                      # last day
        download_csv(data_df=data_tab1.head(144),
                     name="last_day.csv")
    elif data_tab1_range == "last week":                   # last week
        download_csv(data_df=data_tab1.head(1008),
                     name="last_week.csv")
    elif data_tab1_range == "last month":                  # last month
        download_csv(data_df=data_tab1.head(4320),
                     name="last_month.csv")
    elif data_tab1_range == "last year":                   # last year
        download_csv(data_df=data_tab1.head(52560),
                     name="last_year.csv")
    elif data_tab1_range == "all data":                    # all data
        download_csv(data_df=data_tab1,
                     name="all_data.csv")


# tab 2 (Prediction)
with tab2:
    col_date2, col_time2, col_caption2= st.columns([1,1,1])
    # datetime
    date2 = col_date2.date_input("Date:",
                                 value=datetime.datetime(2022,9,30),
                                 min_value=datetime.datetime(2022,9,11),
                                 max_value=datetime.datetime(2022,9,30),
                                 label_visibility="collapsed")
    time2 = col_time2.time_input("Time:",
                                 value=datetime.time(4,30),
                                 step=600,
                                 label_visibility="collapsed")
    col_caption2.caption("👈 Set date and time")
    datetime2 = str(date2)+" "+str(time2)
 
    # proses
    col_button1,col_button2= st.columns([1,2])
    if col_button1.button("Predict ", use_container_width=True,type="primary"):
        col_button2.caption(":green[Predicted!]")
        # klasifikasi
        X_klasifikasi = data_banjir.loc[data_banjir['datetime'] == datetime2].reset_index(drop=True)
        X_klasifikasi = X_klasifikasi[['datetime','height (cm)']].rename(columns={'datetime':'date',
                                                                                  'height (cm)':'height'})
        y_klasifikasi = klasifikasi_banjir(X=X_klasifikasi,
                                 scaler_X=scaler_X_klasifikasi,
                                 model=model_klasifikasi_banjir)
        # prediksi
        X_prediksi = get_X_prediksi(data=data_banjir, 
                                date=datetime2)
        y_pred = prediksi_banjir(data=data_banjir,
                            date=datetime2,
                            X=X_prediksi,
                            scaler_X=scaler_X_prediksi,
                            scaler_y=scaler_y_prediksi,
                            model=model_prediksi_banjir)
        y_pred_status = klasifikasi_banjir(X=y_pred,
                                    scaler_X=scaler_X_klasifikasi,
                                    model=model_klasifikasi_banjir)
        y_pred_status = y_pred_status.rename(columns = {'height':'height_pred (cm)'})

        # info
        with stylable_container(key="container_with_border",
                                css_styles="""{
                                border: 3px solid #dedede;
                                border-radius: 0.5rem;
                                padding: calc(1em - 1px)}"""):
             get_info_banjir2(y_klasifikasi=y_klasifikasi,
                             y_pred_status=y_pred_status)

        # plot grafik    
        st.write("**Flood prediction chart 📈**")

        if y_pred_status.shape[1] == 4:
            data_historical,data_future,data_plot2 = data_plot_tab2(history=X_prediksi, 
                                                                    pred=y_pred_status)

            fig2 = plt.figure(figsize=(10, 4))
            plt.fill_between(x=range(-287,1),y1=data_plot2['height_history (cm)'].iloc[0:288], color='dimgray', alpha=0.4)
            plt.plot(data_plot2['height_history (cm)'].iloc[0:288], color='dimgray', alpha=1, linewidth=3, label='height history')

            plt.fill_between(x=range(1,37),y1=data_plot2['height_true (cm)'].iloc[288:], color='cornflowerblue', alpha=0.4)
            plt.plot(data_plot2['height_true (cm)'].iloc[288:], color='cornflowerblue', alpha=0.7, linewidth=3, label='height true')

            plt.fill_between(x=range(1,37),y1=data_plot2['height_pred (cm)'].iloc[288:], color='red', alpha=0.4)
            plt.plot(data_plot2['height_pred (cm)'].iloc[288:], color='red', alpha=0.7, linewidth=3, label='height pred')

            plt.xlim(-288, 37)
            plt.ylim(0)
            plt.xticks((-288,-252,-216,-180,-144,-108,-72,-36,0,36))
            plt.ylabel('Height (cm)')
            plt.xlabel('Step (1 step = 10 minutes)')
            plt.title('Flood Prediction Chart for the Next 6 Hours', fontweight='bold', fontsize=16, loc='left', pad=25)

            plt.text(-288, -20, s="Datetime : ", fontweight='bold', fontsize=6)
            plt.text(-270, -20, s=datetime2, fontsize=6)
            plt.text(-288, -24, s="Location : ", fontweight='bold', fontsize=6)
            plt.text(-270, -24, s="Padang, Indonesia (-0.955531, 100.477179)", fontsize=6)

            plt.legend(frameon=False, loc=(0,1.01), ncol=3)
            sns.despine(left=True, right=True)
            st.pyplot(fig2)
            
        elif y_pred_status.shape[1] == 2:
            data_historical,data_future,data_plot2 = data_plot_tab2(history=X_prediksi, 
                                                                    pred=y_pred_status)
            
            fig2 = plt.figure(figsize=(10, 4))
            plt.fill_between(x=range(-287,1),y1=data_plot2['height_history (cm)'].iloc[0:288], color='dimgray', alpha=0.4)
            plt.plot(data_plot2['height_history (cm)'].iloc[0:288], color='dimgray', alpha=1, linewidth=3, label='height history')

            plt.fill_between(x=range(1,37),y1=data_plot2['height_pred (cm)'].iloc[288:], color='red', alpha=0.4)
            plt.plot(data_plot2['height_pred (cm)'].iloc[288:], color='red', alpha=0.7, linewidth=3, label='height pred')

            plt.xlim(-288, 37)
            plt.ylim(0)
            plt.xticks((-288,-252,-216,-180,-144,-108,-72,-36,0,36))
            plt.ylabel('Height (cm)')
            plt.xlabel('Step (1 step = 10 minutes)')
            plt.title('Flood Prediction Chart for the Next 6 Hours', fontweight='bold', fontsize=16, loc='left', pad=25)

            plt.text(-288, -20, s="Datetime : ", fontweight='bold', fontsize=6)
            plt.text(-270, -20, s=datetime2, fontsize=6)
            plt.text(-288, -24, s="Location : ", fontweight='bold', fontsize=6)
            plt.text(-270, -24, s="Padang, Indonesia (-0.955531, 100.477179)", fontsize=6)

            plt.legend(frameon=False, loc=(0,1.01), ncol=3)
            sns.despine(left=True, right=True)
            st.pyplot(fig2)
        
        # data
        with st.expander("**Explore prediction data**"):
            st.write("**Historical data:**")
            st.dataframe(data_historical)
            st.write("**Future/predicted data:**")
            st.dataframe(data_future)
        
        # reset button
        "---"      
        st.button("Click to reset")


# tab 3 (Send Message)
with tab3:
    col_date3, col_time3, col_caption3= st.columns([1,1,1])
    # datetime
    date3 = col_date3.date_input("Date: .",
                                 value=datetime.datetime(2022,9,30),
                                 min_value=datetime.datetime(2022,9,11),
                                 max_value=datetime.datetime(2022,9,30),
                                 label_visibility="collapsed")
    time3 = col_time3.time_input("Time: .",
                                 value=datetime.time(4,30),
                                 step=600,
                                 label_visibility="collapsed")
    col_caption3.caption("👈 Set date and time")
    datetime3 = str(date3)+" "+str(time3)

    if st.checkbox("Confirm datetime?"):
        # proses get info
        # klasifikasi
        X_klasifikasi3 = data_banjir.loc[data_banjir['datetime'] == datetime3].reset_index(drop=True)
        X_klasifikasi3 = X_klasifikasi3[['datetime','height (cm)']].rename(columns={'datetime':'date',
                                                                                  'height (cm)':'height'})
        y_klasifikasi3 = klasifikasi_banjir(X=X_klasifikasi3,
                                 scaler_X=scaler_X_klasifikasi,
                                 model=model_klasifikasi_banjir)
        # prediksi
        X_prediksi3 = get_X_prediksi(data=data_banjir, 
                                date=datetime3)
        y_pred3 = prediksi_banjir(data=data_banjir,
                            date=datetime3,
                            X=X_prediksi3,
                            scaler_X=scaler_X_prediksi,
                            scaler_y=scaler_y_prediksi,
                            model=model_prediksi_banjir)
        y_pred_status3 = klasifikasi_banjir(X=y_pred3,
                                    scaler_X=scaler_X_klasifikasi,
                                    model=model_klasifikasi_banjir)
        y_pred_status3 = y_pred_status3.rename(columns = {'height':'height_pred (cm)'})

        # caption
        date_info, height_info, status_info, msg_line1, msg_line2 = get_info_banjir3(y_klasifikasi=y_klasifikasi3,
                                                                                     y_pred_status=y_pred_status3)
        message = f"""
⚠️ *Flood Alert - Prediction info* ⚠️

Datetime: {str(date_info)} WIB
Location: Padang, Indonesia
Height: {str(height_info.round(2))} cm
Status: {status_info}

Message:
- {msg_line1}
- {msg_line2}

---------------------------------------------------
_#StaySafe_
_#QuakeFloodAlert_"""
        
        # image   
        if y_pred_status3.shape[1] == 4:
            data_historical3, data_future3, data_plot3 = data_plot_tab2(history=X_prediksi3, 
                                                                    pred=y_pred_status3)

            fig3 = plt.figure(figsize=(10, 4))
            plt.fill_between(x=range(-287,1),y1=data_plot3['height_history (cm)'].iloc[0:288], color='dimgray', alpha=0.4)
            plt.plot(data_plot3['height_history (cm)'].iloc[0:288], color='dimgray', alpha=1, linewidth=3, label='height history')

            plt.fill_between(x=range(1,37),y1=data_plot3['height_true (cm)'].iloc[288:], color='cornflowerblue', alpha=0.4)
            plt.plot(data_plot3['height_true (cm)'].iloc[288:], color='cornflowerblue', alpha=0.7, linewidth=3, label='height true')

            plt.fill_between(x=range(1,37),y1=data_plot3['height_pred (cm)'].iloc[288:], color='red', alpha=0.4)
            plt.plot(data_plot3['height_pred (cm)'].iloc[288:], color='red', alpha=0.7, linewidth=3, label='height pred')

            plt.xlim(-288, 37)
            plt.ylim(0)
            plt.xticks((-288,-252,-216,-180,-144,-108,-72,-36,0,36))
            plt.ylabel('Height (cm)')
            plt.xlabel('Step (1 step = 10 minutes)')
            plt.title('Flood Prediction Chart for the Next 6 Hours', fontweight='bold', fontsize=16, loc='left', pad=25)

            plt.text(-288, -20, s="Datetime : ", fontweight='bold', fontsize=6)
            plt.text(-270, -20, s=datetime3, fontsize=6)
            plt.text(-288, -24, s="Location : ", fontweight='bold', fontsize=6)
            plt.text(-270, -24, s="Padang, Indonesia (-0.955531, 100.477179)", fontsize=6)

            plt.legend(frameon=False, loc=(0,1.01), ncol=3)
            sns.despine(left=True, right=True)
            
        elif y_pred_status3.shape[1] == 2:
            data_historical,data_future,data_plot3 = data_plot_tab2(history=X_prediksi3, 
                                                                    pred=y_pred_status3)
            
            fig3 = plt.figure(figsize=(10, 4))
            plt.fill_between(x=range(-287,1),y1=data_plot3['height_history (cm)'].iloc[0:288], color='dimgray', alpha=0.4)
            plt.plot(data_plot3['height_history (cm)'].iloc[0:288], color='dimgray', alpha=1, linewidth=3, label='height history')

            plt.fill_between(x=range(1,37),y1=data_plot3['height_pred (cm)'].iloc[288:], color='red', alpha=0.4)
            plt.plot(data_plot3['height_pred (cm)'].iloc[288:], color='red', alpha=0.7, linewidth=3, label='height pred')

            plt.xlim(-288, 37)
            plt.ylim(0)
            plt.xticks((-288,-252,-216,-180,-144,-108,-72,-36,0,36))
            plt.ylabel('Height (cm)')
            plt.xlabel('Step (1 step = 10 minutes)')
            plt.title('Flood Prediction Chart for the Next 6 Hours', fontweight='bold', fontsize=16, loc='left', pad=25)

            plt.text(-288, -20, s="Datetime : ", fontweight='bold', fontsize=6)
            plt.text(-270, -20, s=datetime3, fontsize=6)
            plt.text(-288, -24, s="Location : ", fontweight='bold', fontsize=6)
            plt.text(-270, -24, s="Padang, Indonesia (-0.955531, 100.477179)", fontsize=6)

            plt.legend(frameon=False, loc=(0,1.01), ncol=3)
            sns.despine(left=True, right=True)
        
    # tabs
    tab3_number, tab3_group,= st.tabs(["**Send to phone number(s)**", "**Send to a group**"])
    i=1
    date_name=datetime3.replace(":", "-")
    path_image=f"fig/fig {date_name}.png"

    # tab phone number
    with tab3_number:
        phone_list = st.text_input("**Phone number(s)** (format: +628xxxxxxxxxx, multipe number: split with ','):")
        phone_list = phone_list.replace(" ", "")
        phone = phone_list.split(",")    
        # preview
        if st.checkbox("Preview?"):
            try:
                with stylable_container(key="container_with_border",
                                        css_styles="""{
                                        border: 3px solid #dedede;
                                        border-radius: 0.5rem;
                                        padding: calc(1em - 1px)}"""):
                    st.write(f"**Receiver** : {phone}")
                    st.write("**Image** : ")
                    st.pyplot(fig3)
                    st.write("**Caption** : ")
                    st.write(message)
            except NameError:
                st.error("Please confirm datetime")
        # send message
        if st.button("Send Message", type="primary"):
            try:
                fig3.savefig(fname=path_image, bbox_inches="tight", pad_inches=0.3)
                # for each in phone:
                #     # pywhatkit.sendwhats_image(receiver=each,
                #     #                           img_path=path_image,
                #     #                           caption=message,
                #     #                           wait_time=10,
                #     #                           tab_close=True, 
                #     #                           close_time=3)
                st.success("Message sent!")
            except NameError:
                st.error("Please confirm datetime!")
            st.image(path_image)
    
    # tab group
    # with tab3_group:
    #     group_id = st.text_input("**WhatsApp Group ID:**")
    #     group_id = group_id.replace(" ", "")
    #     # preview
    #     if st.checkbox("Preview? "):
    #         try:
    #             with stylable_container(key="container_with_border",
    #                                     css_styles="""{
    #                                     border: 3px solid #dedede;
    #                                     border-radius: 0.5rem;
    #                                     padding: calc(1em - 1px)}"""):
    #                 st.write(f"**Receiver** : WhatsApp Group, ID: ['{group_id}']")
    #                 st.write("**Image** : ")
    #                 st.pyplot(fig3)
    #                 st.write("**Caption** : ")
    #                 st.write(message)
    #         except NameError:
    #             st.error("Please confirm datetime!")
    #     # send message
    #     if st.button("Send Message ", type="primary"):
    #         try:
    #             fig3.savefig(fname=path_image, bbox_inches="tight", pad_inches=0.3)
    #             pywhatkit.sendwhats_image(receiver=group_id,
    #                                       img_path=path_image, 
    #                                       caption=message,
    #                                       wait_time=10,
    #                                       tab_close=True, 
    #                                       close_time=3)
    #             st.success("Message sent!")
    #         except NameError:
    #             st.error("Please confirm datetime!")


    # # reset button
    # "---"      
    # st.button(" Click to reset   ")
