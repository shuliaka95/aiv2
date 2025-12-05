# fix_firewall.py
import os
import subprocess
import sys

def fix_firewall():
    print("🔧 Настройка фаервола для доступа с других устройств...")
    
    try:
        # Для Linux (ufw)
        if sys.platform.startswith('linux'):
            print("📝 Отключаем фаервол (ufw)...")
            subprocess.run(['sudo', 'ufw', 'disable'], check=True)
            print("✅ Фаервол отключен")
            
        # Для Windows
        elif sys.platform.startswith('win'):
            print("📝 Настраиваем фаервол Windows...")
            subprocess.run([
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                'name=Modulation_Recognition', 'dir=in', 'action=allow',
                'protocol=TCP', 'localport=5000'
            ], check=True)
            print("✅ Правило фаервола добавлено")
            
    except Exception as e:
        print(f"⚠️  Не удалось настроить фаервол автоматически: {e}")
        print("📋 Сделайте вручную:")
        print("   Linux: sudo ufw disable")
        print("   Windows: Разрешите порт 5000 в фаерволе")

def check_port():
    print("\n🔍 Проверка доступности порта 5000...")
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('0.0.0.0', 5000))
        s.close()
        if result == 0:
            print("✅ Порт 5000 открыт и доступен")
        else:
            print("❌ Порт 5000 недоступен")
    except Exception as e:
        print(f"❌ Ошибка проверки порта: {e}")

if __name__ == '__main__':
    fix_firewall()
    check_port()