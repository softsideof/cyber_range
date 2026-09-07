// threat intelligence data — scenario-specific threat actor profiles, CVEs, IOCs
// all data is based on real-world campaigns that these scenarios are modeled after

export interface ThreatActor {
  codename: string;          // e.g. LAZARUS GROUP
  alias: string;             // e.g. APT38
  origin: string;            // e.g. North Korea (DPRK)
  motivation: string;        // e.g. Financial / Espionage
  sectors: string[];         // targeted industries
  knownTools: string[];      // malware / tooling
  mitreGroups: string[];     // MITRE ATT&CK group IDs
  activity: string;          // recent campaign summary
}

export interface CVEEntry {
  cveId: string;
  component: string;
  cvss: number;
  description: string;
  exploitStatus: 'PoC Available' | 'Actively Exploited' | 'Patch Available' | 'No Known Fix';
  active: boolean;           // is this CVE being used in the current simulation
}

export interface IOC {
  type: 'IP' | 'Domain' | 'Hash' | 'Registry Key' | 'File Path' | 'Mutex';
  value: string;
  description: string;
  status: 'active' | 'neutralized' | 'pending';
}

export interface ThreatIntelProfile {
  actor: ThreatActor;
  cves: CVEEntry[];
  iocs: IOC[];
}

// map scenario IDs to their threat intel profiles
const THREAT_INTEL: Record<string, ThreatIntelProfile> = {
  script_kiddie: {
    actor: {
      codename: 'UNKNOWN THREAT ACTOR',
      alias: 'Script Kiddie / Low-Skill',
      origin: 'Unknown — VPN / Tor Exit Node',
      motivation: 'Opportunistic / Credential Theft',
      sectors: ['Unclassified', 'Any Exposed Service'],
      knownTools: ['Hydra', 'Medusa', 'THC-Hydra', 'fail2ban bypass'],
      mitreGroups: [],
      activity: 'Automated SSH brute force campaigns via Tor exit node 185.220.101.42. No targeted intent — scanning entire /16 subnet. Commonly precedes credential stuffing sale on dark-web markets.',
    },
    cves: [
      {
        cveId: 'CVE-2018-15473',
        component: 'OpenSSH < 7.7',
        cvss: 5.3,
        description: 'Username enumeration via timing oracle allows attackers to silently validate SSH user accounts before brute force.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
    ],
    iocs: [
      { type: 'IP', value: '185.220.101.42', description: 'Known Tor exit node — brute force source', status: 'active' },
      { type: 'IP', value: '185.220.101.38', description: 'Related Tor exit node cluster', status: 'pending' },
      { type: 'File Path', value: '/tmp/.ssh-agent-XXXXXX', description: 'Persistent SSH credential cache dropped on compromise', status: 'pending' },
    ],
  },

  phishing_campaign: {
    actor: {
      codename: 'SCATTERED SPIDER',
      alias: 'UNC3944 / 0ktapus',
      origin: 'United Kingdom / United States (Cybercrime)',
      motivation: 'Financial — SIM Swapping, Ransomware, Extortion',
      sectors: ['Technology', 'Finance', 'Retail', 'Gaming'],
      knownTools: ['Raccoon Stealer', 'VidarStealer', 'BlackCat/ALPHV Ransomware', 'Okta admin impersonation'],
      mitreGroups: ['G1015'],
      activity: 'Sophisticated social engineering campaigns targeting IT helpdesk personnel to reset MFA. Known for compromising Okta, Twilio, Cloudflare. Recruits English-speaking teenagers via Telegram. Active since 2022.',
    },
    cves: [
      {
        cveId: 'CVE-2023-4966',
        component: 'Citrix Bleed (NetScaler ADC)',
        cvss: 9.4,
        description: 'Session token disclosure allows unauthenticated remote code execution — bypasses even MFA-protected sessions.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
      {
        cveId: 'CVE-2022-41040',
        component: 'Microsoft Exchange (ProxyNotShell)',
        cvss: 8.8,
        description: 'SSRF vulnerability chained with RCE. Exploited for initial access via phishing email attachments.',
        exploitStatus: 'PoC Available',
        active: false,
      },
    ],
    iocs: [
      { type: 'Domain', value: 'corp-helpdesk-reset[.]com', description: 'Fake IT helpdesk phishing domain', status: 'active' },
      { type: 'Hash', value: 'SHA256: 4a5c8e...f7a2b1', description: 'Raccoon Stealer payload dropped via phishing attachment', status: 'active' },
      { type: 'IP', value: '104.21.88.42', description: 'Phishing kit C2 server (Cloudflare-proxied)', status: 'pending' },
      { type: 'Mutex', value: 'Global\\raccoon_mutex_4f2a', description: 'Raccoon Stealer process mutex', status: 'pending' },
    ],
  },

  apt_lateral_movement: {
    actor: {
      codename: 'COZY BEAR',
      alias: 'APT29 / Midnight Blizzard / The Dukes',
      origin: 'Russia — SVR (Foreign Intelligence Service)',
      motivation: 'Strategic Intelligence Collection / Espionage',
      sectors: ['Government', 'Defence', 'Think Tanks', 'Healthcare', 'Technology'],
      knownTools: ['Cobalt Strike', 'BloodHound', 'mimikatz', 'SUNBURST', 'WellMess', 'PowerShell Empire'],
      mitreGroups: ['G0016'],
      activity: 'Highly sophisticated APT29 is conducting targeted lateral movement after initial foothold via spear-phishing. Known for patient, low-noise operations. Currently targeting Active Directory with credential harvesting using BloodHound and Kerberoasting. Attribution confidence: HIGH.',
    },
    cves: [
      {
        cveId: 'CVE-2020-1472',
        component: 'Zerologon (Windows Netlogon)',
        cvss: 10.0,
        description: 'Critical privilege escalation — unauthenticated attacker can gain Domain Admin in seconds by exploiting Netlogon authentication.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
      {
        cveId: 'CVE-2021-42278',
        component: 'sAMAccountName Spoofing (AD)',
        cvss: 7.5,
        description: 'Active Directory privilege escalation via computer account renaming to impersonate Domain Controllers.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
    ],
    iocs: [
      { type: 'Domain', value: 'dc-sync-azure[.]microsoft-auth[.]io', description: 'C2 domain mimicking Azure AD infrastructure', status: 'active' },
      { type: 'Hash', value: 'SHA256: 8f3c1d...a9e4f2', description: 'mimikatz binary (renamed svchost32.exe)', status: 'active' },
      { type: 'File Path', value: 'C:\\Windows\\System32\\svchost32.exe', description: 'mimikatz disguised as Windows service binary', status: 'active' },
      { type: 'Registry Key', value: 'HKLM\\SYSTEM\\CurrentControlSet\\Services\\WinProxy', description: 'Persistence key for C2 backdoor service', status: 'pending' },
      { type: 'IP', value: '51.83.246.110', description: 'Cobalt Strike Team Server (OVH cloud, NL)', status: 'pending' },
    ],
  },

  ransomware_outbreak: {
    actor: {
      codename: 'LAZARUS GROUP',
      alias: 'APT38 / Hidden Cobra / ZINC',
      origin: 'North Korea — Reconnaissance General Bureau (RGB)',
      motivation: 'Financial — Cryptocurrency theft, Sanctions Evasion',
      sectors: ['Financial Services', 'Healthcare', 'Manufacturing', 'Energy'],
      knownTools: ['WannaCry', 'HERMES', 'FASTCASH', 'Manuscrypt', 'BLINDINGCAN'],
      mitreGroups: ['G0032'],
      activity: 'Lazarus Group deploying WannaCry-variant ransomware leveraging EternalBlue SMB exploit. Campaign specifically targets un-patched Windows 7 / Server 2008 workstations. Cryptocurrency wallets previously linked to $620M Axie Infinity hack. Attribution confirmed by NSA, CISA, and NCSC.',
    },
    cves: [
      {
        cveId: 'CVE-2017-0144',
        component: 'EternalBlue — SMBv1 (Windows)',
        cvss: 9.3,
        description: 'Remote code execution via malformed SMB packets. NSA-developed exploit leaked by Shadow Brokers. Used in WannaCry, NotPetya, and ongoing ransomware campaigns.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
      {
        cveId: 'CVE-2017-0145',
        component: 'EternalRomance — SMBv1 (Windows)',
        cvss: 8.1,
        description: 'Companion exploit to EternalBlue. Provides more reliable RCE against Windows Vista/7/8.1/Server editions.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
    ],
    iocs: [
      { type: 'Hash', value: 'SHA256: ed01eb...3b5bde', description: 'WannaCry ransomware binary — original Shadow Brokers variant', status: 'active' },
      { type: 'Domain', value: 'www.iuqerfsodp9ifjaposdfjhgosurijfaewrwergwea[.]com', description: 'WannaCry kill-switch domain — blocking this activates worm spread', status: 'active' },
      { type: 'IP', value: '197.231.221.211', description: 'WannaCry C2 Tor hidden service exit node', status: 'pending' },
      { type: 'File Path', value: 'C:\\Windows\\tasksche.exe', description: 'WannaCry dropper persistence binary', status: 'active' },
      { type: 'Registry Key', value: 'HKLM\\SOFTWARE\\WanaCrypt0r', description: 'Ransomware configuration registry key', status: 'pending' },
    ],
  },

  supply_chain_compromise: {
    actor: {
      codename: 'COZY BEAR',
      alias: 'APT29 — SolarWinds Cluster',
      origin: 'Russia — SVR',
      motivation: 'Strategic Intelligence / Long-term Access',
      sectors: ['Technology Supply Chain', 'US Government', 'Defence Contractors'],
      knownTools: ['SUNBURST', 'TEARDROP', 'Cobalt Strike', 'RAINDROP', 'GoldMax'],
      mitreGroups: ['G0016'],
      activity: 'APT29 conducted the SolarWinds Orion supply chain attack — inserting SUNBURST backdoor into digitally-signed software updates distributed to 18,000+ organizations. Active undetected for 14 months (Oct 2019 — Dec 2020). Classified as a significant intelligence collection operation against US national security by the Biden administration.',
    },
    cves: [
      {
        cveId: 'CVE-2020-10148',
        component: 'SolarWinds Orion Platform',
        cvss: 9.8,
        description: 'Authentication bypass via URL parameter manipulation allows unauthenticated remote code execution. Used as initial foothold before SUNBURST deployment.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
    ],
    iocs: [
      { type: 'Hash', value: 'SHA256: a25cadd...e4f8c3', description: 'SUNBURST backdoor DLL (SolarWinds.Orion.Core.BusinessLayer.dll)', status: 'active' },
      { type: 'Domain', value: 'avsvmcloud[.]com', description: 'SUNBURST C2 DNS beacon domain — resolves per-victim based on AD domain hash', status: 'active' },
      { type: 'Domain', value: 'deftsecurity[.]com', description: 'TEARDROP secondary payload delivery domain', status: 'pending' },
      { type: 'Registry Key', value: 'HKLM\\SOFTWARE\\SolarWinds\\Orion\\BackupMode', description: 'SUNBURST dormancy configuration key', status: 'pending' },
      { type: 'File Path', value: 'C:\\Windows\\SysWOW64\\netsetupsvc.dll', description: 'RAINDROP Cobalt Strike loader', status: 'pending' },
    ],
  },

  insider_threat_apt: {
    actor: {
      codename: 'SCATTERED SPIDER',
      alias: 'Ghost Operator / UNC3944',
      origin: 'United States / UK — Organized Cybercrime',
      motivation: 'Financial — Data Theft, Extortion, Dark-web Sale',
      sectors: ['Technology', 'Finance', 'MGM/Casino', 'Retail'],
      knownTools: ['Okta phishing', 'AnyDesk', 'Rclone', 'MEGAsync', 'BlackCat/ALPHV'],
      mitreGroups: ['G1015'],
      activity: 'Insider threat variant of Scattered Spider. Recruited executive employee as insider to install AnyDesk remote access tool. Actor then pivoted from exec workstation to cloud email and document vaults. Exfiltrating classified customer data via Rclone to MEGA cloud. Identical TTPs to MGM Resorts $100M breach (2023).',
    },
    cves: [
      {
        cveId: 'CVE-2023-29357',
        component: 'Microsoft SharePoint Server',
        cvss: 9.8,
        description: 'Authentication bypass via spoofed JWT tokens allows privilege escalation to site collection administrator without credentials.',
        exploitStatus: 'PoC Available',
        active: false,
      },
      {
        cveId: 'CVE-2023-23397',
        component: 'Microsoft Outlook (NTLM Hash Leak)',
        cvss: 9.8,
        description: 'Zero-click vulnerability — attacker-controlled calendar reminder triggers automatic NTLM authentication to attacker-controlled server, leaking password hash.',
        exploitStatus: 'Actively Exploited',
        active: true,
      },
    ],
    iocs: [
      { type: 'IP', value: '194.165.16.74', description: 'AnyDesk relay server used by Ghost Operator for remote access', status: 'active' },
      { type: 'Domain', value: 'mega[.]nz/folder/Xxxxxx', description: 'MEGA cloud exfiltration destination', status: 'active' },
      { type: 'File Path', value: 'C:\\Users\\exec\\AppData\\Local\\rclone.exe', description: 'Rclone exfiltration tool installed by insider', status: 'active' },
      { type: 'Hash', value: 'SHA256: 9f2e4a...b7c3d5', description: 'AnyDesk installer with persistence config (silent mode)', status: 'pending' },
    ],
  },
};

export function getThreatIntel(scenarioId: string): ThreatIntelProfile | null {
  return THREAT_INTEL[scenarioId] || null;
}
