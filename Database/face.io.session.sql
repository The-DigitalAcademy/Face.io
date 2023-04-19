DROP TABLE IF EXISTS emp_info;

CREATE TABLE emp_info(
empl_no INT PRIMARY KEY NOT NULL,
full_name VARCHAR(50) NOT NULL,
cohort VARCHAR(25) NOT NULL
);

DROP TABLE IF EXISTS face_embeddings;

CREATE TABLE face_embeddings(
empl_no INT PRIMARY KEY NOT NULL,
embeddings FLOAT[] NOT NULL,
embeddings_type VARCHAR(50) NOT NULL
);


   