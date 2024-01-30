# view.py

import streamlit as st
import sqlite3
import pandas as pd

# Function to fetch data from SQLite database
def fetch_data():
    conn = sqlite3.connect('hotel_booking.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM bookings')
    data = cursor.fetchall()
    
    # Fetching column names
    cursor.execute('PRAGMA table_info(bookings)')
    column_names = [column[1] for column in cursor.fetchall()]
    
    conn.close()
    return column_names, data

# Main Streamlit app
def main():
    st.title('Hotel Bookings')

    # Fetch column names and data from the SQLite database
    column_names, data = fetch_data()

    # Convert data and column names to a Pandas DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # showing the data
    st.write(
        df.style.set_table_styles([{'selector': 'table', 'props': [('width', '100%')]}])
    )

if __name__ == '__main__':
    main()
