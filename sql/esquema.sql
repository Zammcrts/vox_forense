-- Esquema de VOX-Forense (PostgreSQL 16). Lo crea automaticamente la API al arrancar; este archivo es solo documentacion.

CREATE TABLE competencias (
	agent_id VARCHAR(40) NOT NULL, 
	task_type VARCHAR(20) NOT NULL, 
	competencia FLOAT NOT NULL, 
	n_actualizaciones INTEGER NOT NULL, 
	actualizado TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (agent_id, task_type)
);

CREATE TABLE ejecuciones (
	run_id VARCHAR(40) NOT NULL, 
	case_id VARCHAR(200) NOT NULL, 
	modo VARCHAR(10) NOT NULL, 
	estado VARCHAR(15) NOT NULL, 
	nombre_original VARCHAR(300), 
	error TEXT, 
	creado TIMESTAMP WITH TIME ZONE NOT NULL, 
	terminado TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (run_id)
);

CREATE INDEX ix_ejecuciones_case_id ON ejecuciones (case_id);

CREATE TABLE casos (
	case_id VARCHAR(200) NOT NULL, 
	run_id VARCHAR(40) NOT NULL, 
	nombre_original VARCHAR(300), 
	rasgos JSONB NOT NULL, 
	faltantes JSONB NOT NULL, 
	hallazgos JSONB NOT NULL, 
	ontologia JSONB NOT NULL, 
	licitacion JSONB NOT NULL, 
	veredicto VARCHAR(10), 
	regla VARCHAR(4), 
	motivo TEXT, 
	n_indicios INTEGER, 
	n_naturalidad INTEGER, 
	contradictoria BOOLEAN NOT NULL, 
	explicacion TEXT, 
	version_ontologia VARCHAR(32), 
	etiqueta_real VARCHAR(5), 
	creado TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (case_id), 
	UNIQUE (run_id), 
	FOREIGN KEY(run_id) REFERENCES ejecuciones (run_id)
);

CREATE INDEX ix_casos_veredicto ON casos (veredicto);

CREATE TABLE eventos_traza (
	id BIGSERIAL NOT NULL, 
	run_id VARCHAR(40) NOT NULL, 
	seq INTEGER NOT NULL, 
	parent_seq INTEGER, 
	ts FLOAT NOT NULL, 
	evento VARCHAR(20) NOT NULL, 
	agente VARCHAR(40), 
	tarea VARCHAR(10), 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (run_id, seq), 
	FOREIGN KEY(run_id) REFERENCES ejecuciones (run_id)
);

CREATE INDEX ix_eventos_traza_evento ON eventos_traza (evento);

CREATE INDEX ix_eventos_traza_run_id ON eventos_traza (run_id);

CREATE TABLE escalamientos (
	id SERIAL NOT NULL, 
	case_id VARCHAR(200) NOT NULL, 
	motivo TEXT NOT NULL, 
	estado VARCHAR(10) NOT NULL, 
	etiqueta_analista VARCHAR(5), 
	comentario TEXT, 
	creado TIMESTAMP WITH TIME ZONE NOT NULL, 
	resuelto TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	UNIQUE (case_id), 
	FOREIGN KEY(case_id) REFERENCES casos (case_id)
);

CREATE INDEX ix_escalamientos_estado ON escalamientos (estado);
