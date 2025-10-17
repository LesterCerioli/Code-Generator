import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncpg
from schemas import UserRequirements, ArchitectureRequest

logger = logging.getLogger(__name__)

class RequirementsProcessor:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def process_user_requirements(self, user_req: UserRequirements) -> Dict[str, Any]:
        """Process and validate user requirements, extract key information"""
        try:
            business_keywords = self._extract_business_keywords(user_req.business_context)
            technical_requirements = self._analyze_technical_requirements(user_req)
                        
            req_id = await self._store_requirements(user_req, business_keywords, technical_requirements)
            
            return {
                "requirements_id": req_id,
                "business_context_analysis": business_keywords,
                "technical_requirements": technical_requirements,
                "complexity_level": self._assess_complexity(user_req),
                "recommended_architecture_type": self._suggest_architecture_type(user_req)
            }
        except Exception as e:
            logger.error(f"Error processing requirements: {str(e)}")
            raise

    async def _store_requirements(self, user_req: UserRequirements, business_keywords: Dict, tech_reqs: Dict) -> str:
        """Store processed requirements in PostgreSQL with UUID primary key"""
        async with self.db_pool.acquire() as conn:
            query = """
            INSERT INTO public.architecture_requirements 
            (business_context, functional_requirements, non_functional_requirements, 
             expected_scale, integration_points, compliance_requirements, business_keywords,
             technical_analysis, complexity_level, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id::text
            """
            
            req_id = await conn.fetchval(
                query,
                user_req.business_context,
                user_req.functional_requirements,
                user_req.non_functional_requirements,
                user_req.expected_scale,
                user_req.integration_points,
                user_req.compliance_requirements,
                business_keywords,
                tech_reqs,
                self._assess_complexity(user_req),
                datetime.utcnow()
            )
            
            return req_id  

    async def get_requirements_by_id(self, requirements_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve requirements by UUID"""
        async with self.db_pool.acquire() as conn:
            query = """
            SELECT id::text, business_context, functional_requirements, non_functional_requirements,
                   expected_scale, integration_points, compliance_requirements, business_keywords,
                   technical_analysis, complexity_level, created_at
            FROM public.architecture_requirements 
            WHERE id = $1
            """
            
            record = await conn.fetchrow(query, requirements_id)
            if record:
                return dict(record)
            return None

    def _extract_business_keywords(self, business_context: str) -> Dict[str, Any]:
        """Extract key business domains and requirements"""
        keywords = {
            "ecommerce": bool(re.search(r'e-?commerce|online.store|shop|cart|payment', business_context, re.I)),
            "saas": bool(re.search(r'saas|software.as.a.service|subscription', business_context, re.I)),
            "iot": bool(re.search(r'iot|internet.of.things|sensor|device', business_context, re.I)),
            "real_time": bool(re.search(r'real.time|live|streaming|instant', business_context, re.I)),
            "data_analytics": bool(re.search(r'analytics|reporting|dashboard|metrics', business_context, re.I)),
            "mobile": bool(re.search(r'mobile|app|ios|android', business_context, re.I))
        }
        
        return {
            "domains": [k for k, v in keywords.items() if v],
            "has_real_time_requirements": keywords["real_time"],
            "has_mobile_requirements": keywords["mobile"]
        }

    def _analyze_technical_requirements(self, user_req: UserRequirements) -> Dict[str, Any]:
        """Analyze technical aspects from requirements"""
        return {
            "scale_requirements": user_req.expected_scale,
            "security_requirements": len([req for req in user_req.non_functional_requirements 
                                        if 'security' in req.lower() or 'auth' in req.lower()]),
            "performance_requirements": len([req for req in user_req.non_functional_requirements 
                                           if 'performance' in req.lower() or 'speed' in req.lower()]),
            "integration_complexity": len(user_req.integration_points),
            "compliance_level": len(user_req.compliance_requirements)
        }

    def _assess_complexity(self, user_req: UserRequirements) -> str:
        """Assess overall complexity of requirements"""
        complexity_score = (
            len(user_req.functional_requirements) * 2 +
            len(user_req.non_functional_requirements) * 3 +
            len(user_req.integration_points) * 4 +
            len(user_req.compliance_requirements) * 5
        )
        
        if complexity_score > 50:
            return "high"
        elif complexity_score > 20:
            return "medium"
        else:
            return "low"

    def _suggest_architecture_type(self, user_req: UserRequirements) -> str:
        """Suggest initial architecture type based on requirements"""
        if user_req.expected_scale in ["large", "enterprise"]:
            return "microservices"
        elif any('real-time' in req.lower() for req in user_req.non_functional_requirements):
            return "event_driven"
        elif len(user_req.integration_points) > 5:
            return "microservices"
        else:
            return "monolithic"