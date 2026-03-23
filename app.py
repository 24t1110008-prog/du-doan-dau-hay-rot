import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("student_pass_fail_extended_dataset.csv")

print("5 dòng dữ liệu đầu:")
print(data.head())


X = data[['attendance',
          'homework',
          'midterm',
          'final_project',
          'study_hours_per_day',
          'sleep_hours_per_day',
          'social_media_hours',
          'absences',
          'previous_gpa',
          'participation']]

y = data['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Độ chính xác của mô hình:", accuracy)

print("\nBáo cáo phân loại:")

print(classification_report(y_test, y_pred))

print("\n----- Nhập dữ liệu sinh viên -----")
attendance = float(input("Attendance: "))
homework = float(input("Homework: "))
midterm = float(input("Midterm: "))
final_project = float(input("Final project: "))
study_hours = float(input("Study hours per day: "))
sleep_hours = float(input("Sleep hours per day: "))
social_media = float(input("Social media hours: "))
absences = int(input("Absences: "))
previous_gpa = float(input("Previous GPA: "))
participation = float(input("Participation: "))
new_student = [[attendance,
                homework,
                midterm,
                final_project,
                study_hours,
                sleep_hours,
                social_media,
                absences,
                previous_gpa,
                participation]]
prediction = model.predict(new_student)
print("\nKết quả dự đoán:", prediction[0])