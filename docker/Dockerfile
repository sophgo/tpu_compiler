ARG BASE_REPO

FROM $BASE_REPO

LABEL maintainer="charle.hu@cvitek.com"

# Make sure the package repository is up to date.
RUN apt-get update && \
	apt-get install -qy openssh-server && \
	sed -i 's|session    required     pam_loginuid.so|session    optional     pam_loginuid.so|g' /etc/pam.d/sshd && \
	mkdir -p /var/run/sshd && \
	apt-get install -qy openjdk-8-jdk && \
	apt-get install -qy iputils-ping && \
	pip3 install pexpect && \
	apt-get -qy autoremove && \
	useradd -ms /bin/bash jenkins && \
	echo "jenkins:jenkins" | chpasswd
# Standard SSH port
EXPOSE 22
# 启动sshd让jenkins能访问
CMD ["/usr/sbin/sshd", "-D"]