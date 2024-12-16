import functions
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from fpdf import FPDF
import io



st.set_page_config(page_title="Home")
st.title("WhatsApp Chat Analyzer")

chat_format = st.radio(
    "Select WhatsApp Export Format",
    ('iOS', 'Android'),
    help="Choose the format based on your device type"
)

format_type = chat_format.lower()

img_path = "icon1.avif"
img_base64 = functions.img_to_base64(img_path)

# Define CSS style to adjust image size
img_style = """
    max-width: 100%;
    height: auto;
"""

st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" style="{img_style}" class="cover-glow">',
    unsafe_allow_html=True,
)

# Add text below the image with custom styling
st.sidebar.markdown(
    "<p style='font-family: Arial, sans-serif; font-size: 20px; text-align: center; margin-top: 10px;'>WhatsApp Chat Analyzer</p>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

st.sidebar.success("Select a User")

st.markdown(f"### Upload your WhatsApp chat export ({chat_format} format)")


# st.sidebar.title("WhatsApp Chat Analyzer")

#create a file uploaded to upload txt file
uploaded_file = st.file_uploader("Choose a file", help=f"Upload the WhatsApp chat export in {chat_format} format")
if uploaded_file:
    df = functions.create_dataframe(uploaded_file,format_type)
    st.subheader("Overall Data Preview")

    number = st.number_input("Number of rows to view",5)
    st.dataframe(df[df.columns[0:3]].head(number))

    user_list = df['User'].unique().tolist()
    user_list.sort()
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox("",user_list)
    
    if(st.sidebar.button("Show Analysis")):
        #Display basic stats in 4 cols
        num_messages, words, num_media_messages, num_links = functions.fetch_stats(selected_user,df)

        st.subheader(f"Analysis of User: **{selected_user}**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Total Messages</h4>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; font-weight: bold;'>{}</h3>".format(num_messages), unsafe_allow_html=True)
        with col2:
            st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Total Words</h4>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; font-weight: bold;'>{}</h3>".format(words), unsafe_allow_html=True)
        with col3:
            st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Media Shared</h4>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; font-weight: bold;'>{}</h3>".format(num_media_messages), unsafe_allow_html=True)
        with col4:
            st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Links Shared</h4>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; font-weight: bold;'>{}</h3>".format(num_links), unsafe_allow_html=True)

    num_messages, words, num_media_messages, num_links = functions.fetch_stats(selected_user, df)
    

    user_stop_words = st.text_input("Enter words to exclude (comma-separated)")
    if user_stop_words:
        user_stop_words = [word.strip() for word in user_stop_words.split(',')]
    else:
        user_stop_words = []


    st.subheader("Most Common Words - Word Cloud")
    df_wc = functions.top_words(selected_user, df,user_stop_words)
    fig99, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(df_wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig99)

    if selected_user == 'Overall':
        st.subheader("Top Senders Analysis")
        top_senders_df = functions.top_senders(df)
        st.dataframe(top_senders_df)
        

    st.subheader("Monthly Chat Timeline")
    timeline = functions.monthly_timeline(selected_user, df)
    fig1,ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(timeline['Time'], timeline['Message'], color='skyblue')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    st.pyplot(fig1)

    
    # functions.monthly_timeline(selected_user, df)

    show_additional_analysis = st.checkbox("Show Additional Month Analysis")
    if show_additional_analysis:
        # Calculate average number of messages per day
        average_messages_per_month = timeline['Message'].mean()
        st.write("Average messages per month: ", round(average_messages_per_month))

        # Find top 5 busiest days
        top_busiest_months = timeline.nlargest(5, 'Message')
        st.write("Top 5 busiest months:")
        st.write(top_busiest_months)


    st.subheader("Daily Timeline")
    daily_timeline = functions.daily_timeline(selected_user, df)
    fig2, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(daily_timeline['Date'], daily_timeline['Message'], color='lightgreen')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    st.pyplot(fig2)
    
    show_additional_analysis = st.checkbox("Show Additional Day Analysis")
    if show_additional_analysis:
        # Calculate average number of messages per day
        average_messages_per_day = daily_timeline['Message'].mean()
        st.write("Average messages per day: ", round(average_messages_per_day))

        # Find top 5 busiest days
        top_busiest_days = daily_timeline.nlargest(5, 'Message')
        st.write("Top 5 busiest days:")
        st.write(top_busiest_days)

    # activity map
    st.subheader('Activity Map')
    col1,col2,col3 = st.columns(3)
    
    #weekly activity
    with col1:
        st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Busy Weeks</h4>", unsafe_allow_html=True)
        busy_day = functions.week_activity_map(selected_user,df)
        fig3,ax = plt.subplots()
        bars=ax.bar(busy_day.index,busy_day.values,color='purple')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Messages')
        plt.tight_layout()
        st.pyplot(fig3)

    #monthly activity
    with col2:
        st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Busy Months</h4>", unsafe_allow_html=True)
        busy_month = functions.month_activity_map(selected_user, df)
        fig4, ax = plt.subplots()
        bars=ax.bar(busy_month.index, busy_month.values,color='orange')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        plt.tight_layout()
        st.pyplot(fig4)
        
    with col3:
        st.markdown("<h4 style='text-align: center; font-weight: bold; color: green;'>Busy Hours</h4>", unsafe_allow_html=True)
        hourly_activity = functions.hour_activity_map(selected_user, df)
        fig100, ax = plt.subplots()
        bars = ax.bar(hourly_activity.index, hourly_activity.values, color='teal')
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there are messages
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        hour_labels = [f'{i%12 if i%12 != 0 else 12}{" AM" if i < 12 else " PM"}' for i in range(24)]
        plt.xticks(range(24), hour_labels, rotation=45, ha='right')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Messages')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig100)

    # most common words
    st.subheader('Most commmon words')
    most_common_df = functions.common_words(selected_user,df,user_stop_words)
    fig5,ax = plt.subplots()
    bars=ax.barh(most_common_df[0],most_common_df[1])
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width)}',
                ha='left', va='center')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.title('Most Common Words')
    plt.tight_layout()
    st.pyplot(fig5)


    st.subheader("Emoji Analysis")
    emoji_df = functions.emoji_analysis(selected_user,df)

    if emoji_df.shape[0] > 0:
        col1,col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            # Create emoji bar chart using Matplotlib
            fig, ax = plt.subplots(figsize=(8, 8))
            top_emojis = emoji_df.head(10)  # Show top 10 emojis
            bars = ax.bar(top_emojis['emoji'], top_emojis['count'], color='orange')
        
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',fontsize=15)
        
            plt.xlabel('Emoji',fontsize=15)
            plt.ylabel('Count',fontsize=15)
            plt.title("Top 10 Emojis",fontsize=15)
            plt.xticks(rotation=45, ha='right',fontname='Segoe UI Emoji',fontsize=15)
            plt.tight_layout()
        
            # Display the chart using st.pyplot()
            st.pyplot(fig)
    else:
        st.write(":red[No Emojis Sent by this user]")
    
    pdf_report_stream = functions.generate_pdf_report(selected_user,
                                                      df, num_messages,words,
                                                      num_media_messages, num_links, timeline, daily_timeline, 
                                                      busy_day, busy_month, most_common_df, emoji_df,df_wc, user_stop_words)

    st.download_button(label="Download Report",
                       data=pdf_report_stream.getvalue(),
                       file_name=f"whatsapp_chat_analysis_{selected_user}.pdf",
                       mime="application/pdf")
