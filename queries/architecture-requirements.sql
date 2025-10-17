CREATE TABLE architecture_requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_context TEXT NOT NULL,
    functional_requirements JSONB,
    non_functional_requirements JSONB,
    expected_scale VARCHAR(50),
    integration_points JSONB,
    compliance_requirements JSONB,
    business_keywords JSONB,
    technical_analysis JSONB,
    complexity_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);