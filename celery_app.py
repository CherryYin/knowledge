"""
Celery 异步任务配置

用于处理大文件文档摄入的异步任务，避免阻塞 API 请求。

启动 Worker:
    celery -A celery_app worker --loglevel=info --concurrency=4

启动 Flower 监控（可选）:
    celery -A celery_app flower --port=5555

启动 Beat 定时任务（可选）:
    celery -A celery_app beat --loglevel=info
"""

import os
import logging
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Celery 配置
# ─────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "knowledge",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_app"],
)

app.conf.update(
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    
    # 任务路由
    task_routes={
        "celery_app.ingest_document_task": {"queue": "ingestion"},
        "celery_app.daily_summary_task": {"queue": "maintenance"},
    },
    
    # Worker 配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    # 重试配置
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # 结果过期（秒）
    result_expires=3600 * 24,  # 24 小时
    
    # 定时任务
    beat_schedule={
        "daily-summary": {
            "task": "celery_app.daily_summary_task",
            "schedule": crontab(hour=18, minute=0),  # 每天 18:00
        },
    },
)


# ─────────────────────────────────────────────
#  异步任务
# ─────────────────────────────────────────────

@app.task(
    bind=True,
    name="celery_app.ingest_document_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def ingest_document_task(self, file_path: str, kb_id: str = "default", user_id: str = "default"):
    """
    异步文档摄入任务
    
    Args:
        file_path: 文件路径
        kb_id: 知识库 ID
        user_id: 用户 ID
    
    Returns:
        dict: 摄入结果
    """
    import asyncio
    from pathlib import Path
    from ingestion_pipeline import DocumentIngestionPipeline
    
    logger.info(f"[TASK] 开始摄入文档：{file_path}, kb_id={kb_id}, user_id={user_id}")
    
    try:
        # 验证文件存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        # 创建 Pipeline 并执行
        pipeline = DocumentIngestionPipeline()
        result = asyncio.run(pipeline.ingest(file_path, kb_id, user_id))
        
        logger.info(f"[TASK] 文档摄入完成：{result}")
        return {
            "status": "success",
            "result": result,
        }
        
    except FileNotFoundError as e:
        logger.error(f"[TASK] 文件未找到：{e}")
        return {
            "status": "error",
            "error_type": "file_not_found",
            "message": str(e),
        }
        
    except Exception as e:
        logger.exception(f"[TASK] 摄入失败：{e}")
        # 触发重试
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))


@app.task(
    bind=True,
    name="celery_app.daily_summary_task",
    max_retries=1,
)
def daily_summary_task(self):
    """
    每日工作总结任务
    每天 18:00 运行，生成工作日志
    """
    import subprocess
    from datetime import datetime
    
    logger.info("[TASK] 执行每日工作总结")
    
    try:
        script_path = "/home/ubuntu/.openclaw/workspace/cron/daily-summary.sh"
        result = subprocess.run(
            [script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            logger.info(f"[TASK] 每日总结完成：{result.stdout}")
            return {"status": "success", "output": result.stdout}
        else:
            logger.error(f"[TASK] 每日总结失败：{result.stderr}")
            return {"status": "error", "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        logger.error("[TASK] 每日总结脚本超时")
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        logger.exception(f"[TASK] 每日总结异常：{e}")
        return {"status": "error", "error": str(e)}


@app.task(
    bind=True,
    name="celery_app.cleanup_temp_files_task",
    max_retries=1,
)
def cleanup_temp_files_task(self, older_than_hours: int = 24):
    """
    清理临时文件任务
    
    Args:
        older_than_hours: 清理多少小时前的文件
    """
    import subprocess
    from pathlib import Path
    
    logger.info(f"[TASK] 清理 {older_than_hours}h 前的临时文件")
    
    temp_dirs = [
        "/tmp/parse_*",
        "/tmp/uploads/*",
        "/tmp/knowledge_*",
    ]
    
    cleaned_count = 0
    for pattern in temp_dirs:
        try:
            result = subprocess.run(
                ["find", "/tmp", "-name", pattern.split("/")[-1], 
                 "-type", "d" if "*" in pattern else "f",
                 "-mmin", f"+{older_than_hours * 60}",
                 "-delete", "-print"],
                capture_output=True,
                text=True,
            )
            if result.stdout:
                cleaned_count += len(result.stdout.strip().split("\n"))
        except Exception as e:
            logger.warning(f"清理 {pattern} 失败：{e}")
    
    logger.info(f"[TASK] 清理完成，共清理 {cleaned_count} 个文件/目录")
    return {"status": "success", "cleaned_count": cleaned_count}


# ─────────────────────────────────────────────
#  任务状态查询
# ─────────────────────────────────────────────

def get_task_status(task_id: str) -> dict:
    """
    查询任务状态
    
    Args:
        task_id: 任务 ID
    
    Returns:
        dict: 任务状态信息
    """
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id, app=app)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "result": result.result if result.ready() else None,
    }


# ─────────────────────────────────────────────
#  CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 测试任务
    print("Celery App Loaded")
    print(f"Broker: {REDIS_URL}")
    print(f"Registered tasks: {app.tasks}")
