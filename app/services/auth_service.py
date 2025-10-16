import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt  # Agora deve funcionar após a instalação
from dotenv import load_dotenv
from app.config import config
from app.database import db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthTokenService:
    def __init__(self):
        load_dotenv()
        
        self.jwt_secret = os.getenv('JWT_SECRET')
        if not self.jwt_secret or len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET should have 32+ characters")
        
        
        self.client_credentials = self._load_client_credentials()
        
        if not self.client_credentials:
            raise ValueError("CLIENT_ID_1 and SECRET_1 must be set in environment")

    def _load_client_credentials(self) -> Dict[str, str]:
        """Load client credentials from environment variables"""
        credentials = {}
        
        client_id = os.getenv('CLIENT_ID_1')
        secret = os.getenv('SECRET_1')
        
        if client_id and secret:
            credentials[client_id] = secret
        
        return credentials

    def generate_token(self, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Generate JWT token and store it in database"""
        logger.info(f"Iniciando geração de token para client_id={client_id}")

        
        stored_secret = self.client_credentials.get(client_id)
        if not stored_secret:
            logger.warning(f"client_id não encontrado: {client_id}")
            raise ValueError("invalid client_id or secret")
            
        if stored_secret != client_secret:
            logger.warning(f"Segredo inválido para client_id={client_id}")
            raise ValueError("invalid client_id or secret")

        
        expiration = datetime.now(timezone.utc) + timedelta(minutes=2)
        logger.debug(f"Expiração definida para: {expiration.isoformat()}")

        # Create JWT token with same claims as Go service
        payload = {
            "client_id": client_id,
            "exp": expiration
        }

        try:
            
            token_string = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            logger.info(f"Token JWT gerado com sucesso para client_id={client_id}")
        except Exception as e:
            logger.error(f"Falha ao assinar token JWT: {e}")
            raise ValueError(f"erro ao gerar token: {e}")

        # Store token in database using parameterized query
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO public.auth_tokens 
                        (id, client_id, jwt_token, expires_at, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        str(uuid.uuid4()),
                        client_id,
                        token_string,
                        expiration,
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc)
                    ))
                    conn.commit()
                    
            logger.info(f"Token salvo no banco com sucesso para client_id={client_id}")
            
        except Exception as e:
            logger.error(f"Falha ao salvar token no banco de dados: {e}")
            raise ValueError(f"error to save on database: {e}")

        return {
            "token": token_string,
            "client_id": client_id,
            "expires_at": expiration.isoformat()
        }

    def validate_token(self, token_string: str) -> bool:
        """Validate JWT token against database and cryptographic signature"""
        try:
            # First check if token exists and is not expired in database
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT EXISTS (
                            SELECT 1 FROM public.auth_tokens 
                            WHERE jwt_token = %s AND expires_at > NOW()
                        ) AS is_valid
                    ''', (token_string,))
                    
                    result = cursor.fetchone()
                    is_valid = result['is_valid'] if result else False

            if not is_valid:
                return False

            # Then validate JWT cryptographic signature
            jwt.decode(token_string, self.jwt_secret, algorithms=["HS256"])
            return True

        except jwt.ExpiredSignatureError:
            logger.warning("Token JWT expirado")
            return False
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token JWT inválido: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro ao validar token: {e}")
            return False

    def get_valid_token(self, client_id: str) -> Optional[str]:
        """Get valid non-expired token for client_id"""
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT jwt_token FROM public.auth_tokens 
                        WHERE client_id = %s AND expires_at > NOW() 
                        ORDER BY created_at DESC LIMIT 1
                    ''', (client_id,))
                    
                    result = cursor.fetchone()
                    return result['jwt_token'] if result else None
                    
        except Exception as e:
            logger.error(f"Erro ao buscar token válido: {e}")
            return None

    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens from database, return number of deleted tokens"""
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        DELETE FROM public.auth_tokens 
                        WHERE expires_at <= NOW()
                    ''')
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
            if deleted_count > 0:
                logger.info(f"Limpeza de tokens expirados: {deleted_count} tokens removidos")
                
            return deleted_count
            
        except Exception as e:
            logger.error(f"Erro ao limpar tokens expirados: {e}")
            return 0


# Global instance
auth_token_service = AuthTokenService()