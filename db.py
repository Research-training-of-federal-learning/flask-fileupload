import sqlite3

conn = sqlite3.connect('db.db')
cur = conn.cursor()
# cur.execute('create table USER(USERNAME TEXT KEY NOT NULL, PASSWORD TEXT NOT NULL)')
# cur_data = cur.execute('SELECT USERNAME, PASSWORD FROM USER')

# sql = 'SELECT 1 FROM USER WHERE USERNAME=? AND PASSWORD=?'
sql = 'SELECT * FROM USER'
is_valid_user = cur.execute(sql, ()).fetchone()
# is_valid_user = cur.execute(sql, ('nemo', '1234')).fetchone()
print(is_valid_user)

conn.close()