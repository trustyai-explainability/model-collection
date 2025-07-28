FROM registry.access.redhat.com/ubi8/ubi-minimal:latest

WORKDIR /mnt/data

COPY --chown=1000:1000 data/ /mnt/data/

USER 1000

CMD ["/bin/sh"]
