import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectionResult:
    """检测结果数据模型"""
    id: Optional[int] = None
    image_url: str = ""
    image_hash: str = ""
    detection_time: datetime = None
    locks_detected: int = 0
    unlocked_locks: int = 0
    lock_positions: str = ""  # JSON格式存储锁的位置信息
    confidence_score: float = 0.0
    dingtalk_message_id: str = ""
    user_id: str = ""
    group_id: str = ""
    is_safe: bool = True
    
    def __post_init__(self):
        if self.detection_time is None:
            self.detection_time = datetime.now()


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "lock_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建检测结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_url TEXT NOT NULL,
                    image_hash TEXT UNIQUE NOT NULL,
                    detection_time DATETIME NOT NULL,
                    locks_detected INTEGER DEFAULT 0,
                    unlocked_locks INTEGER DEFAULT 0,
                    lock_positions TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    dingtalk_message_id TEXT,
                    user_id TEXT,
                    group_id TEXT,
                    is_safe BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建锁位置详情表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lock_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER,
                    lock_type TEXT,
                    is_locked BOOLEAN,
                    confidence REAL,
                    position_x INTEGER,
                    position_y INTEGER,
                    width INTEGER,
                    height INTEGER,
                    FOREIGN KEY (detection_id) REFERENCES detection_results (id)
                )
            ''')
            
            # 创建模型训练记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    training_start DATETIME NOT NULL,
                    training_end DATETIME,
                    epochs INTEGER,
                    batch_size INTEGER,
                    learning_rate REAL,
                    train_loss REAL,
                    val_loss REAL,
                    map_score REAL,
                    accuracy REAL,
                    model_path TEXT,
                    dataset_size INTEGER,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_detection_result(self, result: DetectionResult) -> int:
        """保存检测结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO detection_results 
                (image_url, image_hash, detection_time, locks_detected, unlocked_locks, 
                 lock_positions, confidence_score, dingtalk_message_id, user_id, group_id, is_safe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.image_url,
                result.image_hash,
                result.detection_time,
                result.locks_detected,
                result.unlocked_locks,
                result.lock_positions,
                result.confidence_score,
                result.dingtalk_message_id,
                result.user_id,
                result.group_id,
                result.is_safe
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            return result_id
    
    def save_lock_details(self, detection_id: int, lock_details: List[Dict]):
        """保存锁的详细信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for detail in lock_details:
                cursor.execute('''
                    INSERT INTO lock_details 
                    (detection_id, lock_type, is_locked, confidence, 
                     position_x, position_y, width, height)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_id,
                    detail.get('lock_type', 'unknown'),
                    detail.get('is_locked', False),
                    detail.get('confidence', 0.0),
                    detail.get('position_x', 0),
                    detail.get('position_y', 0),
                    detail.get('width', 0),
                    detail.get('height', 0)
                ))
            
            conn.commit()
    
    def get_detection_history(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """获取检测历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detection_results 
                ORDER BY detection_time DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                if result['lock_positions']:
                    result['lock_positions'] = json.loads(result['lock_positions'])
                results.append(result)
            
            return results
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Dict]:
        """根据ID获取检测结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detection_results WHERE id = ?
            ''', (detection_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                result = dict(zip(columns, row))
                if result['lock_positions']:
                    result['lock_positions'] = json.loads(result['lock_positions'])
                return result
            return None
    
    def save_training_record(self, record: Dict) -> int:
        """保存训练记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_records 
                (model_name, training_start, training_end, epochs, batch_size, 
                 learning_rate, train_loss, val_loss, map_score, model_path, 
                 dataset_size, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.get('model_name', ''),
                record.get('training_start', datetime.now()),
                record.get('training_end'),
                record.get('epochs'),
                record.get('batch_size'),
                record.get('learning_rate'),
                record.get('train_loss'),
                record.get('val_loss'),
                record.get('map_score'),
                record.get('model_path'),
                record.get('dataset_size'),
                record.get('status', 'pending')
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            return record_id
    
    def update_training_record(self, record_id: int, updates: Dict):
        """更新训练记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values())
            values.append(record_id)
            
            cursor.execute(f'''
                UPDATE training_records 
                SET {set_clause}
                WHERE id = ?
            ''', values)
            
            conn.commit()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总检测次数
            cursor.execute('SELECT COUNT(*) FROM detection_results')
            total_detections = cursor.fetchone()[0]
            
            # 不安全的检测次数
            cursor.execute('SELECT COUNT(*) FROM detection_results WHERE is_safe = FALSE')
            unsafe_detections = cursor.fetchone()[0]
            
            # 总锁数量
            cursor.execute('SELECT SUM(locks_detected) FROM detection_results')
            total_locks = cursor.fetchone()[0] or 0
            
            # 未锁数量
            cursor.execute('SELECT SUM(unlocked_locks) FROM detection_results')
            total_unlocked = cursor.fetchone()[0] or 0
            
            # 今日检测次数
            cursor.execute('''
                SELECT COUNT(*) FROM detection_results 
                WHERE DATE(detection_time) = DATE('now')
            ''')
            today_detections = cursor.fetchone()[0]
            
            return {
                'total_detections': total_detections,
                'unsafe_detections': unsafe_detections,
                'total_locks': total_locks,
                'total_unlocked': total_unlocked,
                'today_detections': today_detections,
                'safety_rate': (total_detections - unsafe_detections) / total_detections * 100 if total_detections > 0 else 100
            }


# 全局数据库实例
db_manager = DatabaseManager()