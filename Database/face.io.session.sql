DROP TABLE IF EXISTS emp_info;

CREATE TABLE emp_info(
empl_no INT PRIMARY KEY NOT NULL,
full_name VARCHAR(50) NOT NULL,
surname VARCHAR(50) NOT NULL,
cohort VARCHAR(25) NOT NULL,
event_time TIMESTAMP   
);



   