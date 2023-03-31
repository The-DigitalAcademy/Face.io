DROP TABLE IF EXISTS Faceio;

CREATE TABLE Faceio(
empl_no INT PRIMARY KEY NOT NULL,
full_name VARCHAR(50) NOT NULL,
surname VARCHAR(50) NOT NULL,
cohort VARCHAR(25) NOT NULL,
event_time TIME   
);


INSERT INTO Faceio(empl_no,full_name,surname,cohort,event_time )
VALUES
    (0755,'Gudani','Mbedzi', 'Data Science','07:30'),
    (0765, 'Sibongile', 'Maluleka', 'Data Science','07:30')
   