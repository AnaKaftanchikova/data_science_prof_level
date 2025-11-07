import moduls.data_loader as data_loader
import moduls.log as log

from notebooks import clustering, umap_lda, vae

if __name__ == "__main__":
    logger = log.setup_logger()
    logger.info("=== Запуск проекта ===")

    logger.info("Кластеризация")
    clustering.run()

    logger.info("Методы понижения размерности (UMAP, LDA)")
    umap_lda.run()

    logger.info("Вариационный автоэнкодер (VAE)")
    vae.run()

    logger.info("Всё завершено успешно!")