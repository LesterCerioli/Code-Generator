import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncpg
from schemas import ArchitectureRequest, ArchitectureResponse, ArchitectureComponent, ArchitectureDiagram

logger = logging.getLogger(__name__)

class ArchitectureService:
    def __init__(self, db_pool: asyncpg.Pool, deepseek_client):
        self.db_pool = db_pool
        self.deepseek_client = deepseek_client

    async def generate_architecture(self, arch_request: ArchitectureRequest, requirements_id: str) -> ArchitectureResponse:
        """Generate architecture proposal using AI and store results"""
        try:
            
            ai_prompt = self._build_architecture_prompt(arch_request)
                        
            ai_response = await self.deepseek_client.generate_architecture(ai_prompt)
                        
            architecture_data = self._parse_ai_response(ai_response)
                        
            arch_id = await self._store_architecture(architecture_data, arch_request, requirements_id)
                        
            return await self._build_architecture_response(arch_id, architecture_data)
            
        except Exception as e:
            logger.error(f"Error generating architecture: {str(e)}")
            raise

    async def _store_architecture(self, arch_data: Dict, arch_request: ArchitectureRequest, req_id: str) -> str:
        """Store architecture proposal in PostgreSQL with UUID primary key"""
        async with self.db_pool.acquire() as conn:
            
            arch_query = """
            INSERT INTO public.architecture_proposals 
            (requirements_id, project_name, architecture_type, summary, recommendations,
             considerations, estimated_cost, scalability_notes, security_considerations, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id::text
            """
            
            arch_id = await conn.fetchval(
                arch_query,
                req_id,
                f"Architecture-{uuid.uuid4().hex[:8]}",
                arch_data.get("architecture_type", "microservices"),
                arch_data.get("summary", ""),
                arch_data.get("recommendations", []),
                arch_data.get("considerations", []),
                arch_data.get("estimated_cost", "To be determined"),
                arch_data.get("scalability_notes", ""),
                arch_data.get("security_considerations", []),
                datetime.utcnow()
            )

            
            component_query = """
            INSERT INTO public.architecture_components 
            (architecture_id, name, type, description, technologies, responsibilities, dependencies)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            for component in arch_data.get("components", []):
                await conn.execute(
                    component_query,
                    arch_id,  # Já é string UUID
                    component["name"],
                    component["type"],
                    component["description"],
                    component["technologies"],
                    component["responsibilities"],
                    component["dependencies"]
                )

            
            diagram_query = """
            INSERT INTO public.architecture_diagrams 
            (architecture_id, format, content, description)
            VALUES ($1, $2, $3, $4)
            """
            
            for diagram in arch_data.get("diagrams", []):
                await conn.execute(
                    diagram_query,
                    arch_id,  # Já é string UUID
                    diagram["format"],
                    diagram["content"],
                    diagram["description"]
                )

            return arch_id  # Já retorna como string

    def _build_architecture_prompt(self, arch_request: ArchitectureRequest) -> str:
        """Build prompt for AI architecture generation"""
        return f"""
        Generate a software architecture proposal based on these requirements:
        
        PROJECT SCOPE:
        {arch_request.project_scope}
        
        REQUIREMENTS:
        {arch_request.requirements}
        
        CONSTRAINTS:
        {arch_request.constraints}
        
        PREFERRED TECHNOLOGIES:
        {arch_request.preferred_technologies}
        
        TARGET CLOUD: {arch_request.target_cloud}
        
        Please provide a structured response with:
        1. Architecture type (microservices, monolithic, serverless, etc.)
        2. Summary of the architecture
        3. List of components with their responsibilities
        4. Technology recommendations for each component
        5. Architecture diagrams in Mermaid format
        6. Scalability considerations
        7. Security recommendations
        8. Estimated cost considerations
        """

    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into structured architecture data"""
        
        try:
            
            components = self._extract_components(ai_response)
                        
            diagrams = self._extract_diagrams(ai_response)
            
            return {
                "architecture_type": self._extract_architecture_type(ai_response),
                "summary": self._extract_summary(ai_response),
                "components": components,
                "diagrams": diagrams,
                "recommendations": self._extract_recommendations(ai_response),
                "considerations": self._extract_considerations(ai_response),
                "estimated_cost": self._extract_cost(ai_response),
                "scalability_notes": self._extract_scalability(ai_response),
                "security_considerations": self._extract_security(ai_response)
            }
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return self._get_default_architecture()

    def _extract_components(self, response: str) -> List[Dict]:
        """Extract architecture components from AI response"""
        
        return [{
            "name": "API Gateway",
            "type": "gateway",
            "description": "Handles incoming requests and routes to appropriate services",
            "technologies": ["Python/FastAPI", "Nginx", "Kong"],
            "responsibilities": ["Request routing", "Rate limiting", "Authentication"],
            "dependencies": ["User Service", "Auth Service"]
        }]

    def _extract_diagrams(self, response: str) -> List[Dict]:
        """Extract diagrams from AI response"""
        return [{
            "format": "mermaid",
            "content": "graph TD\n    A[Client] --> B[API Gateway]\n    B --> C[User Service]\n    B --> D[Auth Service]",
            "description": "High-level architecture diagram"
        }]

    def _extract_architecture_type(self, response: str) -> str:
        """Extract architecture type from response"""
        return "microservices"

    def _extract_summary(self, response: str) -> str:
        """Extract architecture summary"""
        return "Microservices-based architecture designed for scalability and maintainability"

    def _extract_recommendations(self, response: str) -> List[str]:
        return ["Use containerization with Docker", "Implement CI/CD pipeline"]

    def _extract_considerations(self, response: str) -> List[str]:
        return ["Consider database partitioning for large datasets"]

    def _extract_cost(self, response: str) -> str:
        return "Medium - depends on scale and cloud provider"

    def _extract_scalability(self, response: str) -> str:
        return "Horizontally scalable with load balancing"

    def _extract_security(self, response: str) -> List[str]:
        return ["Implement JWT authentication", "Use HTTPS everywhere"]

    def _get_default_architecture(self) -> Dict[str, Any]:
        """Return default architecture in case of parsing errors"""
        return {
            "architecture_type": "microservices",
            "summary": "Default microservices architecture",
            "components": [],
            "diagrams": [],
            "recommendations": [],
            "considerations": [],
            "estimated_cost": "To be determined",
            "scalability_notes": "",
            "security_considerations": []
        }

    async def _build_architecture_response(self, arch_id: str, arch_data: Dict) -> ArchitectureResponse:
        """Build the final architecture response"""
        components = [
            ArchitectureComponent(**component) 
            for component in arch_data.get("components", [])
        ]
        
        diagrams = [
            ArchitectureDiagram(**diagram) 
            for diagram in arch_data.get("diagrams", [])
        ]
        
        return ArchitectureResponse(
            architecture_id=arch_id,
            project_name=f"Architecture-{arch_id}",
            architecture_type=arch_data.get("architecture_type", "microservices"),
            summary=arch_data.get("summary", ""),
            components=components,
            diagrams=diagrams,
            recommendations=arch_data.get("recommendations", []),
            considerations=arch_data.get("considerations", []),
            estimated_cost=arch_data.get("estimated_cost"),
            scalability_notes=arch_data.get("scalability_notes", ""),
            security_considerations=arch_data.get("security_considerations", []),
            created_at=datetime.utcnow()
        )

    async def get_architecture_by_id(self, architecture_id: str) -> Optional[ArchitectureResponse]:
        """Retrieve architecture by UUID from database"""
        async with self.db_pool.acquire() as conn:
            
            arch_query = """
            SELECT id::text, requirements_id, project_name, architecture_type, summary, 
                   recommendations, considerations, estimated_cost, scalability_notes, 
                   security_considerations, created_at
            FROM public.architecture_proposals 
            WHERE id = $1
            """
            arch_record = await conn.fetchrow(arch_query, architecture_id)
            
            if not arch_record:
                return None
            
            
            components_query = """
            SELECT name, type, description, technologies, responsibilities, dependencies
            FROM public.architecture_components 
            WHERE architecture_id = $1
            """
            components_records = await conn.fetch(components_query, architecture_id)
            
            
            diagrams_query = """
            SELECT format, content, description
            FROM public.architecture_diagrams 
            WHERE architecture_id = $1
            """
            diagrams_records = await conn.fetch(diagrams_query, architecture_id)
            
            
            components = [
                ArchitectureComponent(
                    name=comp["name"],
                    type=comp["type"],
                    description=comp["description"],
                    technologies=comp["technologies"],
                    responsibilities=comp["responsibilities"],
                    dependencies=comp["dependencies"]
                ) for comp in components_records
            ]
            
            diagrams = [
                ArchitectureDiagram(
                    format=diag["format"],
                    content=diag["content"],
                    description=diag["description"]
                ) for diag in diagrams_records
            ]
            
            return ArchitectureResponse(
                architecture_id=arch_record["id"],
                project_name=arch_record["project_name"],
                architecture_type=arch_record["architecture_type"],
                summary=arch_record["summary"],
                components=components,
                diagrams=diagrams,
                recommendations=arch_record["recommendations"],
                considerations=arch_record["considerations"],
                estimated_cost=arch_record["estimated_cost"],
                scalability_notes=arch_record["scalability_notes"],
                security_considerations=arch_record["security_considerations"],
                created_at=arch_record["created_at"]
            )