import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string
import emoji
from collections import Counter
from PIL import Image
from urlextract import URLExtract
import base64
import os 
from fpdf import FPDF
import io
import tempfile
import shutil
from datetime import datetime




@st.cache_data(show_spinner=False)
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def create_dataframe(file,chat_format='ios'):
    # Read the content of the file as text
    file_content = file.getvalue().decode('utf-8-sig')

    # Define regular expression pattern to extract timestamp, user, and message
    ios_pattern = r'^\[(\d{2}/\d{2}/\d{2}),\s(\d{2}:\d{2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.*)$'
    android_pattern = r'^(\d{2}/\d{2}/\d{2}),\s(\d{1,2}:\d{2}\s[ap]m)\s-\s([^:]+):\s(.*)$'

    pattern=ios_pattern if chat_format=='ios' else android_pattern

    # Initialize lists to store extracted information
    timestamps = []
    users = []
    messages = []

    # Iterate over each line in the file content
    for line in file_content.split('\n'):
        match = re.match(pattern, line)
        # if match:
        #     timestamps.append(match.group(1) + " " + match.group(2))  # Combine date and time
        #     # Clean the user name to handle special characters
        if match:
            date = match.group(1)
            time = match.group(2)

            if chat_format=='android':
                try:
                    time_obj = datetime.strptime(time, '%I:%M %p')
                    time = time_obj.strftime('%I:%M:%S %p')
                except ValueError:
                    continue
            timestamps.append(f"{date} {time}")

            user_name = match.group(3).strip().lstrip('~').strip()
            users.append(user_name)
            messages.append(match.group(4))
    
    # Create DataFrame from the extracted information
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'User': users,
        'Message': messages
    })

    # Convert 'Timestamp' column to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y %I:%M:%S %p')

    # Extract new columns for date, time (in 24-hour format), and month
    df['Date'] = df['Timestamp'].dt.date
    df['Time'] = df['Timestamp'].dt.strftime('%H:%M:%S')
    df['Month'] = df['Timestamp'].dt.month

    df['Month_name'] = df['Timestamp'].dt.month_name()
    df['day'] = df['Timestamp'].dt.day
    df['day_name'] = df['Timestamp'].dt.day_name()
    df['hour'] = df['Timestamp'].dt.hour
    df['year'] = df['Timestamp'].dt.year
    # Print the DataFrame
    return df


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]  
    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['Message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['Message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    num_links = df['Message'].apply(lambda x: len(extract.find_urls(x)))

    return num_messages, len(words), num_media_messages, num_links.sum()

def remove_punctuation(message):
    x = re.sub('[%s]'% re.escape(string.punctuation), '', message)
    return x


def top_words(selected_user,df,user_stop_words):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # Download stop words if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    
    # Add custom stop words
    custom_stops = {'media', 'omitted', 'message', 'deleted'}
    stop_words.update(custom_stops)
    stop_words.update(set((user_stop_words)))
    temp = df[df['User'] != 'group_notification']
    temp = temp[temp['Message'] != '<Media omitted>\n']

    words = []
    for message in temp['Message']:
        # Convert to lowercase and remove punctuation
        clean_message = remove_punctuation(message.lower())
        
        # Split into words and filter out stop words and single characters
        message_words = [word for word in clean_message.split() 
                        if word not in stop_words 
                        and len(word) > 1]
        words.extend(message_words)

    # Create string of all words
    words_string = ' '.join(words)

    # Generate word cloud
    wc = WordCloud(width=800,
                  height=400,
                  min_font_size=10,
                  background_color='white',
                  colormap='viridis',  # You can try different colormaps like 'plasma', 'inferno', etc.
                  max_words=100)
    
    df_wc = wc.generate(words_string)
    
    # Create figure with no axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(df_wc, interpolation='bilinear')
    ax.axis('off')  # Hide axes
    plt.tight_layout(pad=0)  # Remove padding
    
    return df_wc



def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    timeline = df.groupby(['year', 'Month', 'Month_name']).count()['Message'].reset_index()
    
    # Create month-year format
    timeline['Time'] = timeline['Month_name'] + '-' + timeline['year'].astype(str)
    
    # Create the figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(timeline['Time'], timeline['Message'], color='skyblue')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Rotate labels slightly for better readability
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Month-Year')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    
    return timeline
    

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get daily message counts
    daily_timeline = df.groupby('Date').count()['Message'].reset_index()
    
    # Remove the first data point
    daily_timeline = daily_timeline.iloc[1:]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(daily_timeline['Date'], 
                 daily_timeline['Message'], 
                 color='lightgreen',
                 width=0.8)  # Adjust bar width
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')

    
    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # Get counts and sort by custom day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['day_name'].value_counts()
    counts = counts.reindex(day_order,fill_value=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color='purple')

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Messages')
    plt.tight_layout()

    return counts

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    Month_order = ['January', 'February', 'March', 'April', 'May', 'June','July','August','September','October','November','December']
    counts = df['Month_name'].value_counts()
    counts = counts.reindex(Month_order,fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color='purple')

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Month')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    return counts



def hour_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # Get counts for each hour (0-23)
    hour_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
    
    return hour_counts


def common_words(selected_user, df,user_stop_words):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # Download stop words if not already downloaded
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(set(user_stop_words))
    
    temp = df[df['User'] != 'group_notification']
    temp = temp[temp['Message'] != '<Media omitted>\n']
    
    words = []
    
    for message in temp['Message']:
        # Convert to lowercase and remove punctuation
        clean_message = remove_punctuation(message.lower())
        
        # Split into words and filter out stop words
        message_words = [word for word in clean_message.split() if word not in stop_words]
        words.extend(message_words)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    most_common_df = most_common_df.iloc[::-1] 
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(most_common_df[0], most_common_df[1], color='green')
    
    # Add count labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center')
    
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.title('Most Common Words')
    plt.tight_layout()
    
    return most_common_df


def top_senders(df):
    # Calculate message counts per user
    sender_counts = df['User'].value_counts()
    
    # Calculate percentages
    total_messages = df.shape[0]
    sender_percentages = (sender_counts / total_messages * 100).round(2)
    
    # Create DataFrame with counts and percentages
    top_senders_df = pd.DataFrame({
        'User': sender_counts.index,
        'Messages': sender_counts.values,
        'Percentage': sender_percentages.values
    })
    
    return top_senders_df
    



def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', 
                values='message', aggfunc='count').fillna(0)
    return user_heatmap


extract = URLExtract()



#func will only work in group chat analysis
def active_users(df):
    x = df['User'].value_counts().head()
    df = round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x,df


def emoji_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    emojis = []
    for message in df['Message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    emoji_df = emoji_df.rename(columns={0: "emoji", 1: "count"})
    return emoji_df



def clean_text_for_pdf(text):
    """
    Clean text for PDF encoding by replacing unsupported characters
    """
    try:
        # Try to encode the text using latin-1, replacing unsupported characters
        return text.encode('latin-1', 'replace').decode('latin-1')
    except UnicodeEncodeError:
        # If that fails, remove or replace problematic characters
        cleaned_text = ''.join(char if ord(char) < 128 else '?' for char in text)
        return cleaned_text



def generate_pdf_report(selected_user, df, num_messages, words, num_media_messages, num_links, timeline, daily_timeline, busy_day, busy_month, most_common_df, emoji_df, word_cloud_image, user_stop_words):
    pdf = FPDF()
    temp_dir = tempfile.mkdtemp()

    # Function to center image on page
    def add_centered_image(image_path, y_position=None, width=190):
        page_width = pdf.w
        if y_position is not None:
            pdf.set_y(y_position)
        x_position = (page_width - width) / 2
        pdf.image(image_path, x=x_position, y=pdf.get_y(), w=width)

    pdf.add_page()

    # Title Section
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, clean_text_for_pdf("WhatsApp Chat Analysis Report"), ln=True, align="C")
    pdf.ln(5)
    
    # Summary Box with border - centered
    box_height = 40
    box_width = 190
    box_x = (pdf.w - box_width) / 2
    pdf.rect(box_x, pdf.get_y(), box_width, box_height)
    pdf.set_xy(box_x + 5, pdf.get_y() + 5)
    pdf.set_font("Arial", size=12)
    
    # Add summary content
    current_x = box_x + 5
    pdf.set_x(current_x)
    pdf.cell(0, 8, clean_text_for_pdf(f"User: {selected_user}"), ln=True)
    pdf.set_x(current_x)
    pdf.cell(box_width/2 - 5, 8, clean_text_for_pdf(f"Total Messages: {num_messages}"), 0)
    pdf.cell(box_width/2 - 5, 8, clean_text_for_pdf(f"Total Words: {words}"), 0, ln=True)
    pdf.set_x(current_x)
    pdf.cell(box_width/2 - 5, 8, clean_text_for_pdf(f"Media Shared: {num_media_messages}"), 0)
    pdf.cell(box_width/2 - 5, 8, clean_text_for_pdf(f"Links Shared: {num_links}"), 0, ln=True)

    # Top Senders Analysis (if Overall view)
    if selected_user == 'Overall':
        pdf.ln(15)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Top Senders Analysis", ln=True, align='C')
        
        # Calculate table dimensions and position
        table_width = 160  # Reduced from 190 for better centering
        col1_width = 80
        col2_width = 40
        col3_width = 40
        table_x = (pdf.w - table_width) / 2
        
        sender_counts = df['User'].value_counts().head(10)
        total_messages = df.shape[0]
        sender_percentages = (sender_counts / total_messages * 100).round(2)
        
        # Create centered table with alternating background colors
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        
        # Headers
        pdf.set_x(table_x)
        pdf.cell(col1_width, 10, "User", 1, 0, 'C', True)
        pdf.cell(col2_width, 10, "Messages", 1, 0, 'C', True)
        pdf.cell(col3_width, 10, "Percentage", 1, 1, 'C', True)
        
        # Data rows
        pdf.set_font("Arial", size=12)
        fill = False
        for user, count in sender_counts.items():
            pdf.set_x(table_x)
            percentage = sender_percentages[user]
            clean_user = user.encode('ascii', 'replace').decode('ascii').strip('~')
            pdf.cell(col1_width, 10, clean_user, 1, 0, 'L', fill)
            pdf.cell(col2_width, 10, str(count), 1, 0, 'C', fill)
            pdf.cell(col3_width, 10, f"{percentage:.2f}%", 1, 1, 'C', fill)
            fill = not fill

    # Word Analysis Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Word Analysis", ln=True, align='C')
    
    # Generate and save word cloud
    wordcloud = top_words(selected_user, df, user_stop_words)
    wordcloud_plot = os.path.join(temp_dir, 'wordcloud.png')
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(wordcloud_plot, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Add centered word cloud
    add_centered_image(wordcloud_plot, pdf.get_y(), 170)
    
    # Add common words bar chart
    pdf.ln(85)
    pdf.cell(0, 10, "Most Common Words", ln=True, align='C')
    fig_words, ax_words = plt.subplots(figsize=(12, 4))
    most_common_df = common_words(selected_user, df, user_stop_words)
    bars = ax_words.barh(most_common_df[0], most_common_df[1], color='green')
    for bar in bars:
        width = bar.get_width()
        ax_words.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width)}', ha='left', va='center')
    plt.tight_layout()
    
    words_plot = os.path.join(temp_dir, 'common_words.png')
    fig_words.savefig(words_plot, format='png', bbox_inches='tight', dpi=300)
    add_centered_image(words_plot, pdf.get_y())
    plt.close(fig_words)

    # Timeline Analysis Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Message Timeline Analysis", ln=True, align='C')
    
    # Monthly timeline
    pdf.cell(0, 10, "Monthly Timeline", ln=True, align='C') 
    fig_monthly, ax_monthly = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.8)
    bars = ax_monthly.bar(timeline['Time'], timeline['Message'], color='skyblue')
    for bar in bars:
        height = bar.get_height()
        ax_monthly.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    monthly_plot = os.path.join(temp_dir, 'monthly_timeline.png')
    fig_monthly.savefig(monthly_plot, format='png', bbox_inches='tight', dpi=300)
    add_centered_image(monthly_plot, pdf.get_y(),170)
    plt.close(fig_monthly)
    
    # Daily timeline
    pdf.ln(85)
    pdf.cell(0, 10, "Daily Timeline", ln=True, align='C')
    fig_daily, ax_daily = plt.subplots(figsize=(12, 6))
    bars = ax_daily.bar(daily_timeline['Date'], daily_timeline['Message'], color='lightgreen')
    for bar in bars:
        height = bar.get_height()
        ax_daily.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    daily_plot = os.path.join(temp_dir, 'daily_timeline.png')
    fig_daily.savefig(daily_plot, format='png', bbox_inches='tight', dpi=300)
    add_centered_image(daily_plot, pdf.get_y())
    plt.close(fig_daily)

    # Activity Patterns Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Activity Patterns", ln=True, align='C')
    
    # Weekly and Monthly activity
    pdf.cell(0, 10, "Weekly and Monthly Activity", ln=True, align='C')
    fig_activity, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Weekly activity
    bars1 = ax1.bar(busy_day.index, busy_day.values, color='purple')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    ax1.set_title('Weekly Activity')
    ax1.tick_params(axis='x', rotation=45)
    
    # Monthly activity
    bars2 = ax2.bar(busy_month.index, busy_month.values, color='orange')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    ax2.set_title('Monthly Activity')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    activity_plot = os.path.join(temp_dir, 'activity_patterns.png')
    fig_activity.savefig(activity_plot, format='png', bbox_inches='tight', dpi=300)
    add_centered_image(activity_plot, pdf.get_y(),170)
    plt.close(fig_activity)
    
    # Hourly activity
    pdf.ln(85)
    pdf.cell(0, 10, "Hourly Activity", ln=True, align='C')
    fig_hour, ax_hour = plt.subplots(figsize=(12, 6))
    hourly_activity = hour_activity_map(selected_user, df)
    bars = ax_hour.bar(hourly_activity.index, hourly_activity.values, color='teal')
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_hour.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    hour_labels = [f'{h%12 or 12}{["AM", "PM"][h//12]}' for h in hourly_activity.index]
    ax_hour.set_xticks(hourly_activity.index)
    ax_hour.set_xticklabels(hour_labels, rotation=45, ha='right')
    ax_hour.set_title('Hourly Activity')
    plt.tight_layout()
    
    hourly_plot = os.path.join(temp_dir, 'hourly_activity.png')
    fig_hour.savefig(hourly_plot, format='png', bbox_inches='tight', dpi=300)
    add_centered_image(hourly_plot, pdf.get_y())
    plt.close(fig_hour)

    # Emoji Analysis Section (if emojis exist)
    if not emoji_df.empty:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Emoji Analysis", ln=True, align='C')
        
        # Create emoji visualization
        fig_emoji, ax_emoji = plt.subplots(figsize=(16, 8))
        top_emojis = emoji_df.head(10)

        plt.subplots_adjust(bottom=0.6)

        bars = ax_emoji.bar(top_emojis['emoji'], top_emojis['count'], color='orange')
        
        for bar in bars:
            height = bar.get_height()
            ax_emoji.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=15)
        
        plt.xlabel('Emoji', fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.title("Top 10 Emojis", fontsize=15)
        plt.xticks(rotation=45, ha='right', fontsize=15)
        plt.tight_layout()
        
        emoji_plot = os.path.join(temp_dir, 'emoji_analysis.png')
        fig_emoji.savefig(emoji_plot, format='png', bbox_inches='tight',dpi=300)
        plt.close(fig_emoji)
        add_centered_image(emoji_plot, pdf.get_y(),170)
        
        
        # Add emoji statistics table
        pdf.ln(85)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Emoji Statistics", ln=True, align='C')
        
        # Create centered table with alternating colors
        table_width = pdf.w / 2
        col_width = table_width / 2
        table_x = (pdf.w - table_width) / 2
        row_height = 10
        pdf.set_fill_color(240, 240, 240)
        
        # Headers
        pdf.set_x(table_x)
        pdf.cell(col_width, row_height, "Emoji", 1, 0, 'C', True)
        pdf.cell(col_width, row_height, "Count", 1, 1, 'C', True)
        
        # Data
        pdf.set_font("Arial", size=12)
        fill = False
        for _, row in top_emojis.iterrows():
            pdf.set_x(table_x)
            pdf.cell(col_width, row_height, clean_text_for_pdf(row['emoji']), 1, 0, 'C', fill)
            pdf.cell(col_width, row_height, str(row['count']), 1, 1, 'C', fill)
            fill = not fill

    # Create the final PDF
    pdf_stream = io.BytesIO()
    pdf_output_string = pdf.output(pdf_stream, dest='S')
    pdf_stream.write(pdf_output_string.encode('latin-1'))
    pdf_stream.seek(0)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    return pdf_stream
