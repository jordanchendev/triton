def get_queue_length(queue_name: str) -> int:
    """Get the number of pending messages in a Celery queue via Redis."""
    try:
        from triton.workers.celery_app import celery_app
        with celery_app.connection_or_acquire() as conn:
            return conn.default_channel.queue_declare(queue_name, passive=True).message_count
    except Exception:
        return 0
