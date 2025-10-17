CREATE TABLE architecture_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requirements_id UUID REFERENCES architecture_requirements(id),
    project_name VARCHAR(255),
    architecture_type VARCHAR(50),
    summary TEXT,
    recommendations JSONB,
    considerations JSONB,
    estimated_cost VARCHAR(100),
    scalability_notes TEXT,
    security_considerations JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE architecture_components (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    architecture_id UUID REFERENCES architecture_proposals(id),
    name VARCHAR(100),
    type VARCHAR(50),
    description TEXT,
    technologies JSONB,
    responsibilities JSONB,
    dependencies JSONB
);

CREATE TABLE architecture_diagrams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    architecture_id UUID REFERENCES architecture_proposals(id),
    format VARCHAR(20),
    content TEXT,
    description TEXT
);