class Solution:
    def validIPAddress(self, IP: str) -> str:
        if "." in IP:
            parts = IP.split(".")
            if len(parts)!=4:
                return "Neither"
            
            for part in parts:
                if part.isalpha():
                    return "Neither"
                
                if part.startswith('0') and len(part) > 1:
                    return "Neither"
                
                try:
                    if int(part) > 255 or int(part) < 0:
                        return "Neither"
                except:
                    return "Neither"
            
            return "IPv4"
        
        if ":" in IP:
            parts = IP.split(":")
            
            if len(parts) != 8:
                return "Neither"
            
            for part in parts:
                if not part.isalnum():
                    return "Neither"
                
                if len(part) > 4:
                    return "Neither"
                
                try:
                    num = int(part, 16)
                except:
                    return "Neither"
            return "IPv6"
        
        else:
            return "Neither"
