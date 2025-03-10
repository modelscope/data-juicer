import threading
import time
from typing import Callable, Dict


class EventDrivenMixin:
    """Mixin for event-driven capabilities in operations.

    This mixin provides functionality for registering event handlers,
    triggering events, and managing event polling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_handlers = {}
        self.polling_threads = {}
        self.stop_polling_flags = {}

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def trigger_event(self, event_type: str, data: Dict):
        """Trigger an event and call all registered handlers.

        Args:
            event_type: Type of event to trigger
            data: Event data to pass to handlers
        """
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(data)

    def start_polling(self,
                      event_type: str,
                      poll_func: Callable,
                      interval: int = 60):
        """Start polling for a specific event type.

        Args:
            event_type: Type of event to poll for
            poll_func: Function to call for polling
            interval: Polling interval in seconds
        """
        if event_type in self.polling_threads and self.polling_threads[
                event_type].is_alive():
            return

        self.stop_polling_flags[event_type] = False

        def polling_worker():
            while not self.stop_polling_flags.get(event_type, False):
                try:
                    result = poll_func()
                    if result:
                        self.trigger_event(event_type, result)
                except Exception as e:
                    import logging
                    logging.error(f'Error in polling for {event_type}: {e}')
                time.sleep(interval)

        thread = threading.Thread(target=polling_worker, daemon=True)
        self.polling_threads[event_type] = thread
        thread.start()

    def stop_polling(self, event_type: str):
        """Stop polling for a specific event type.

        Args:
            event_type: Type of event to stop polling for
        """
        if event_type in self.polling_threads:
            self.stop_polling_flags[event_type] = True
            self.polling_threads[event_type].join(timeout=1)
            del self.polling_threads[event_type]

    def stop_all_polling(self):
        """Stop all polling threads."""
        event_types = list(self.polling_threads.keys())
        for event_type in event_types:
            self.stop_polling(event_type)

    def wait_for_completion(self,
                            condition_func: Callable[[], bool],
                            timeout: int = 3600,
                            poll_interval: int = 10,
                            error_message: str = 'Operation timed out'):
        """Wait for a condition to be met.

        Args:
            condition_func: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
            error_message: Error message to raise on timeout

        Raises:
            TimeoutError: If the condition is not met within the timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(poll_interval)

        raise TimeoutError(error_message)


class NotificationMixin:
    """Mixin for sending notifications through various channels.

    This mixin provides functionality for sending notifications via email,
    Slack, DingTalk, and other platforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notification_config = kwargs.get('notification_config', {})
        self.notification_handlers = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'dingtalk': self._send_dingtalk_notification,
        }

    def send_notification(self,
                          message: str,
                          notification_type: str = None,
                          **kwargs):
        """Send a notification through the specified channel.

        Args:
            message: The message to send
            notification_type: The type of notification to send (email, slack, dingtalk)
                               If None, will use all configured notification types
            **kwargs: Additional parameters specific to the notification type

        Returns:
            bool: Whether the notification was sent successfully
        """  # noqa: E501
        if notification_type is None:
            # Send to all configured notification types
            success = True
            for ntype in self.notification_config.keys():
                if ntype in self.notification_handlers:
                    success = success and self.notification_handlers[ntype](
                        message, **kwargs)
            return success
        elif notification_type in self.notification_handlers:
            return self.notification_handlers[notification_type](message,
                                                                 **kwargs)
        else:
            import logging
            logging.error(
                f'Unsupported notification type: {notification_type}')
            return False

    def _send_email_notification(self, message: str, **kwargs):
        """Send an email notification.

        Args:
            message: The message to send
            **kwargs: Additional parameters for email configuration
                      (recipients, subject, etc.)

        Returns:
            bool: Whether the email was sent successfully
        """
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            config = self.notification_config.get('email', {})

            # Override config with kwargs if provided
            smtp_server = kwargs.get('smtp_server', config.get('smtp_server'))
            smtp_port = kwargs.get('smtp_port', config.get('smtp_port', 587))
            sender_email = kwargs.get('sender_email',
                                      config.get('sender_email'))
            sender_password = kwargs.get('sender_password',
                                         config.get('sender_password'))
            recipients = kwargs.get('recipients', config.get('recipients', []))
            subject = kwargs.get(
                'subject',
                config.get('subject', 'Notification from Data Juicer'))

            if (not smtp_server or not sender_email or not sender_password
                    or not recipients):
                import logging
                logging.error('Missing required email configuration')
                return False

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            return True
        except Exception as e:
            import logging
            logging.error(f'Failed to send email notification: {e}')
            return False

    def _send_slack_notification(self, message: str, **kwargs):
        """Send a Slack notification.

        Args:
            message: The message to send
            **kwargs: Additional parameters for Slack configuration
                      (webhook_url, channel, etc.)

        Returns:
            bool: Whether the notification was sent successfully
        """
        try:
            import json

            import requests

            config = self.notification_config.get('slack', {})

            # Override config with kwargs if provided
            webhook_url = kwargs.get('webhook_url', config.get('webhook_url'))
            channel = kwargs.get('channel', config.get('channel'))
            username = kwargs.get('username',
                                  config.get('username', 'Data Juicer'))

            if not webhook_url:
                import logging
                logging.error('Missing required Slack webhook URL')
                return False

            # Prepare payload
            payload = {
                'text': message,
                'username': username,
            }

            if channel:
                payload['channel'] = channel

            # Send notification
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'})

            return response.status_code == 200
        except Exception as e:
            import logging
            logging.error(f'Failed to send Slack notification: {e}')
            return False

    def _send_dingtalk_notification(self, message: str, **kwargs):
        """Send a DingTalk notification.

        Args:
            message: The message to send
            **kwargs: Additional parameters for DingTalk configuration
                      (access_token, secret, etc.)

        Returns:
            bool: Whether the notification was sent successfully
        """
        try:
            import base64
            import hashlib
            import hmac
            import json
            import time
            import urllib.parse

            import requests

            config = self.notification_config.get('dingtalk', {})

            # Override config with kwargs if provided
            access_token = kwargs.get('access_token',
                                      config.get('access_token'))
            secret = kwargs.get('secret', config.get('secret'))

            if not access_token:
                import logging
                logging.error('Missing required DingTalk access token')
                return False

            # Prepare URL with signature if secret is provided
            url = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}'  # noqa: E501

            if secret:
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f'{timestamp}\n{secret}'
                hmac_code = hmac.new(secret.encode(),
                                     string_to_sign.encode(),
                                     digestmod=hashlib.sha256).digest()
                sign = urllib.parse.quote_plus(
                    base64.b64encode(hmac_code).decode())
                url = f'{url}&timestamp={timestamp}&sign={sign}'

            # Prepare payload
            payload = {'msgtype': 'text', 'text': {'content': message}}

            # Send notification
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'})

            result = response.json()
            return result.get('errcode') == 0
        except Exception as e:
            import logging
            logging.error(f'Failed to send DingTalk notification: {e}')
            return False
