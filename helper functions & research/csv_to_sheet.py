import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(credentials)

spreadsheet = client.open('csv_to_sheet')
worksheet = spreadsheet.sheet1  # assuming the data goes into the first sheet

with open(f'attendance/attendance_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv', 'r') as file_obj:
    content = file_obj.read()
    rows = content.split('\n')
    data = [row.split(',') for row in rows if row]  # remove empty rows and split the values by comma
    worksheet.append_rows(data)
