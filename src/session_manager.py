import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from models import ChatMessage

logger = logging.getLogger(__name__)

@dataclass
class Session:
    """会话对象"""
    id: str
    created_at: datetime
    last_accessed: datetime
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

class SessionManager:
    """会话管理器"""
    
    def __init__(self, timeout: int = 3600, max_active: int = 100):
        self.sessions: Dict[str, Session] = {}
        self.timeout = timeout
        self.max_active = max_active
    
    def create_session(self, session_id: Optional[str] = None) -> Session:
        """创建新会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # 检查活跃会话数量
        active_count = sum(1 for s in self.sessions.values() if s.active)
        if active_count >= self.max_active:
            # 清理过期会话
            self._cleanup_expired_sessions()
            
            # 再次检查
            active_count = sum(1 for s in self.sessions.values() if s.active)
            if active_count >= self.max_active:
                # 清理最老的会话
                oldest_session = min(
                    [s for s in self.sessions.values() if s.active],
                    key=lambda s: s.last_accessed
                )
                oldest_session.active = False
                logger.info(f"由于达到最大会话数，关闭会话 {oldest_session.id}")
        
        now = datetime.now()
        session = Session(
            id=session_id,
            created_at=now,
            last_accessed=now
        )
        
        self.sessions[session_id] = session
        logger.info(f"创建新会话: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            # 检查是否过期
            if self._is_session_expired(session):
                session.active = False
                logger.info(f"会话 {session_id} 已过期")
                return None
            
            # 更新最后访问时间
            session.last_accessed = datetime.now()
            return session
        
        return None
    
    def update_session(self, session_id: str, message: ChatMessage):
        """更新会话消息"""
        session = self.get_session(session_id)
        if session:
            session.messages.append(message)
            session.last_accessed = datetime.now()
            logger.debug(f"更新会话 {session_id}，添加消息: {message.role}")
    
    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """获取会话消息历史"""
        session = self.get_session(session_id)
        if session:
            return session.messages
        return []
    
    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: {session_id}")
    
    def _is_session_expired(self, session: Session) -> bool:
        """检查会话是否过期"""
        if not session.active:
            return True
        
        expire_time = session.last_accessed + timedelta(seconds=self.timeout)
        return datetime.now() > expire_time
    
    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
    
    def get_active_sessions_count(self) -> int:
        """获取活跃会话数量"""
        return sum(1 for s in self.sessions.values() if s.active)
    
    def get_all_sessions(self) -> List[Session]:
        """获取所有会话"""
        return list(self.sessions.values())
