from data.mock_db import mock_students

def search_user_data(user_id):
    for student in mock_students:
        if student["id"] == user_id:
            return student
    return None