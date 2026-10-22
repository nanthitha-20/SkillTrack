import sqlite3

conn = sqlite3.connect('data/database/skilltrack.db')
c = conn.cursor()

c.execute('SELECT * FROM submissions')
rows = c.fetchall()
print('Current Submissions in DB:', rows)

c.execute("DELETE FROM submissions WHERE code LIKE '%Hello from SkillTrack Intern Portal!%'")
deleted = c.rowcount
print(f'Deleted {deleted} test submission record(s).')

c.execute('SELECT count(*) FROM submissions')
print('Submissions remaining in DB:', c.fetchone()[0])

c.execute('PRAGMA foreign_key_check;')
fk = c.fetchall()
print('PRAGMA foreign_key_check:', fk)

conn.commit()
conn.close()
