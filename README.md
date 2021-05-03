# The Pigeonator

An AI driven pigeon scarer using a Raspberry Pi.

## Background
We have a pigeon problem. It's not just that these birds are so fat they can barely fly (the incredulity of seeing one take off brings back memories of watching "Dumbo" as a child or my first trip on a 747). Nor is it the piles of pigeon cr\*p on the walls and garden furniture or the incessant cooing that we hear down our bedroom chimney at 6am in the morning. No, for me, the final straw was the pond. 

A couple of years ago we build a small pond in the garden. It certainly attracts wildlife, which was the aim. There are definitely more (pleasant) insects and birds in the garden now and they are joy to watch. Unfortunately, the previously resident pigeons have taken to performing their daily ablutions in there too. I have no idea why, or where they've been, but every pigeon bath leaves a revolting and most unsightly slick of oil on the surface of the water. I could stand it no longer and something had to be done.

## Pigeonator Mk I

Despite their unpleasantness, I'm not one for causing the pigeons particular harm; I just want to scare them off. My first attempt was the purchase of a *Pest XT Jet Spray* from Amazon, which is a battery operated motion detector that connects to the garden hose. This emits a 5s spray of water over a 60 degree arc whenever motion is detected in the general area. It certainly worked and did keep the pigeons away. No more oil slicks! However, a couple of issues prevented it from being the complete end to the story:

* The motion detector would sense movement from trees and plants blowing in the wind. The constant triggering in these circumstances was (probably) an irritation to the neigbours and a waste of water. On windy days I'd therefore have to nip out and turn the sensitivity down. Of course, I'd also then have to remember to turn it back up - on several occasions the reminder to do this was the appearence of another slick on the surface of the pond.
* More importantly, perhaps, the detector was indiscriminate and *also scared away the very birds we wanted to encourage*.

A more specific sensor trigger was required.

## The Rise, Fall and Rise of AI

I did quite a lot of work with neural networks back in the 1980's. In those days, they were limited in effectiveness because the computers that could train them were low powered (IBM PCs - no GPUs) and there was usually a dearth of data available. That, coupled with the fact that business uses were restricted (because the networks could never explain *why* they did something) meant that my interest gradually waned.

Scroll forward to 2019 when my eldest son started a PhD in Machine Learning at the University of Bath. Now, he told me, AI and neural networks are back. It's just that today they are called *Deep Learning*. Also, now there is a ton of data and processing power available everywhere so they might actually be useful for something (you can see where I'm going with this, I think). They still can't explain the reasoning behind their decisionmaking but we'll not let that deter us; after all, if we squirt the neighbours cat rather than a pigeon, we won't lose that much sleep over it.

## Pigeonator Mk II

So the new plan is to create a pigeon scaring device that is sensitive only to these fat birds and not to other cute wildlife and the vagaries of the weather. Perhaps we can use AI image classification to do this? With the advent of TensorFlow (and other ML toolkits) it seems this should be possible and, what's more, this stuff can even run on a mobile, low powered, computer like a Raspberry Pi. Now we're talking! 

## Pi Setup

For use with the HQ camera, you need to bump the GPU memory in /boot/config.txt to 176Mb (https://www.raspberrypi.org/forums/viewtopic.php?t=278381)

* Install IFTTT webhook package:
  ```bash
  pip3 install git+https://github.com/DrGFreeman/IFTTT-Webhook.git
  ```

## ML Server Setup

We're using a Debian 10 (Buster) VM running under Proxmox. Configured with 8Gb memory and 32Gb disk.

* Passthrough host CPU features (without this TensorFlow may throw an Illegal instruction)
  Edit the VM CONF file and add `args: -cpu host,kvm=off`
  ```bash
  sudo nano /etc/pve/qemu-server/XXX.conf
  ```

* Allow sudo access for user
  ```bash
  su root
  /usr/sbin/usermod -aG sudo bower
  ```

* Install QEMU Guest Agent
  ```bash 
  apt-get install qemu-guest-agent
  ```
  Make sure that the agent is enabled in the Proxmox VM options and reboot.

* Install XRDP
  ```bash
  sudo apt update
  sudo apt install xfce4 xfce4-goodies xorg dbus-x11 x11-xserver-utils
  sudo apt install xrdp 
  sudo adduser xrdp ssl-cert
  ```

* Find IP address and setup Remote Desktop login
  ```bash
  ip -c a
  ```

* Fix up XRDP if required (Authentication Required to Create Color Managed Device)
  See this blog post: https://c-nergy.be/blog/?p=12073
  ```bash
  sudo sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config

  sudo bash -c "cat >/etc/polkit-1/localauthority/50-local.d/45-allow.colord.pkla" <<EOF
  [Allow Colord all Users]
  Identity=unix-user:*
  Action=org.freedesktop.color-manager.create-device;org.freedesktop.color-manager.create-profile;org.freedesktop.color-manager.delete-device;org.freedesktop.color-manager.delete-profile;org.freedesktop.color-manager.modify-device;org.freedesktop.color-manager.modify-profile
  ResultAny=no
  ResultInactive=no
  ResultActive=yes
  EOF

  sudo bash -c "cat >/etc/polkit-1/localauthority/50-local.d/46-allow-update-repo.pkla" <<EOF
  [Allow Package Management all Users]
  Identity=unix-user:*
  Action=org.freedesktop.packagekit.system-sources-refresh
  ResultAny=yes
  ResultInactive=yes
  ResultActive=yes
  EOF
  ```

* Install VSCode
  Goto  https://code.visualstudio.com/Download and download the appropriate .deb installer.
  ```bash 
  sudo apt install ./code_1.55.2-1618307277_amd64.deb
  ```

* Install Git & Keyring (for GitHub authentication) and fetch Pigeonator Repo
  ```bash
  sudo apt install git # DON'T use git-all - it will hang at boot
  sudo apt install gnome-keyring
  git config --global user.name "Andy Bower"
  git config --global user.email "bower@object-arts.com"

* Set up Samba for file sharing
  Follow basic instructions here: https://vitux.com/debian_samba/

  ```bash
  sudo apt install samba
  sudo mkdir /samba
  sudo chmod 777 /samba
  sudo cp /etc/samba/smb.conf ~/Documents smb_backup.conf
  sudo nano /etc/samba/smb.conf

  # Add to bottom of file:
  [samba-share]
  comment = Samba on Debian
  path = /samba
  read-only = no
  browsable = yes
  writeable = yes
  valid_users = samba, bower

  sudo useradd samba
  sudo smbpasswd -a samba
  sudo smbpasswd -a bower
  sudo systemctl restart smbd.service
  ```

* Set up Python
  ```bash
  sudo apt-get install python3-venv
  ```

* Set up Firewall
  ```bash
  sudo apt install ufw
  sudo ufw default deny incoming
  sudo ufw default allow outgoing
  sudo ufw allow 38100 # Or whatever
  sudo ufw enable
  ```

* Fetch Pigeonator
  mkdir Projects
  cd Projects
  git clone https://github.com/bowerhaus/Pigeonator.git
  ```
