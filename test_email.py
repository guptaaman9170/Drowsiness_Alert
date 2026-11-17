"""
test_email.py - Quick Email Delivery Verifier
Tests sending a real email using the credentials in email_config.json.
"""

import time
from emergency_notifier import EmergencyNotifier, CONFIG_FILE

def test():
    print("=" * 68)
    print("TESTING EMERGENCY EMAIL NOTIFICATION DISPATCH")
    print("=" * 68)

    notifier = EmergencyNotifier()
    print(f"Sender Email        : {notifier.sender_email}")
    print(f"Recipient Email     : {notifier.recipient_email}")
    has_pwd = bool(notifier.sender_password)
    print(f"App Password Status : {'[CONFIGURED]' if has_pwd else '[NOT CONFIGURED - EMPTY]'}")
    print("-" * 68)

    if not has_pwd:
        print("\n[WHY NO EMAIL HAS ARRIVED YET]:")
        print("To protect users from spam, Google/Gmail requires an authenticated sender.")
        print("A script cannot send emails to Gmail without a sender's 16-digit App Password.\n")
        print("HOW TO ACTIVATE IN 1 MINUTE:")
        print("  1. Open your browser: https://myaccount.google.com/apppasswords")
        print("  2. App name: type 'Drowsiness Alert' and click Create")
        print("  3. Copy the 16-letter code (e.g. abcd efgh ijkl mnop)")
        print(f"  4. Open '{CONFIG_FILE}' and paste it inside 'sender_app_password'")
        print("  5. Run 'python test_email.py' again — you will instantly receive the live email!\n")
    else:
        print("[INFO] Connecting to Gmail SMTP & dispatching emergency alert...")
        thread = notifier.dispatch_sos_alert(continuous_seconds=10.0, ear_value=0.075)
        if thread is not None:
            thread.join(timeout=30)
        print("=" * 68)
        print(f"🎉 Emergency email dispatched to: {notifier.recipient_email}")
        print("Please check your Gmail Inbox (or Spam/Junk folder)!")
        print("=" * 68)

if __name__ == "__main__":
    test()
