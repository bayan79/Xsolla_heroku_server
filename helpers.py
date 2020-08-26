import datetime 
import os
import platform


class Notifier:
    """ Simple OS notifier to copy-paste-use 
    Example:
        notifier = Notifier()
        notifier.msg("Hello", title='Notifier')
    """
    def __init__(self):
        self.os = platform.system()
        
    def msg(self, message, title='Notifier'):
        """Send message to OS"""
        message = message.replace('\'', ' ').replace('\"', ' ')
        if self.os == 'Linux':
            os.system(f"notify-send \"{title}\" '{message}'")
        elif self.os == 'Darwin':
            os.system(f"osascript -e \'display notification \"{message}\" with title \"{title}\"\'")
        else:
            # TODO: win notification
            pass


class Clock:
    """Simple clock to copy-paste-use
    Example:
        >>> clock = Clock("Request")
        >>> data = requests.get('http://facebuk.com/users/all/secret/100TB/download')
        >>> clock.tik("Loaded! ")
        [  Request  ]Loaded!    123456.789 s
        >>>
    """
    def __init__(self, label, notify_os=False):
        self.label = label
        self.start = datetime.datetime.now()
        self.notifier = Notifier() if notify_os else None
    
    def reset(self):
        """Reset time"""
        self.start = datetime.datetime.now()
        
    def tik(self, output='', show_time=False):
        """Reset time and output"""
        now = datetime.datetime.now()
        self.start, delta = now, now - self.start
        message = f"[{self.label:^10}] {output:15}" 
        if show_time:
            message += f"{delta.total_seconds():12.2f} s"
            
        print(message)
        if self.notifier:
            self.notifier.msg(message, title="Python Clock")
        

if __name__ == "__main__":
    clock = Clock('Test')
    for i in range(10000000):
        pass
    clock.tik('test run', show_time=True)
