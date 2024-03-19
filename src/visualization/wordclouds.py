import matplotlib.pyplot as plt

def Rakuten_txt_wordcloud(data, token_col_name, categories):
    plt_rows = len(categories)
    plt_idx = 0
    fig, axs = plt.subplots(plt_rows, 1, figsize=(12, 6*plt_rows))
    stopwords = set(STOPWORDS)
    for code in categories.index:
        img = Image.open('./wordcloud-masks/console.jpg')
        mask_coloring = np.array(img)
        wordcloud = WordCloud(
                        background_color='white',
                        mask=mask_coloring,
                        min_font_size=5,
                        max_font_size=30,
                        contour_width=1,
                        random_state=42,
                        max_words=4000,
                        stopwords=stopwords,
                    ).generate(' '.join(data[data.prdtypecode == code][token_col_name]))
        # img_colors=ImageColorGenerator(mask_coloring)
        # axs[plt_idx].imshow(wordcloud.recolor(color_func=img_colors), interpolation="bilinear")
        wc_img = Image.fromarray(wordcloud.to_array())
        # back_img= img.resize(wc_img.size)
        # img_new = Image.alpha_composite(back_img, wc_img)
        axs[plt_idx].imshow(wc_img)
        axs[plt_idx].set_title(str(code) + ' ' + categories.loc[code][0])
        axs[plt_idx].axis("off")
        plt_idx += 1

    plt.tight_layout()
    plt.show()


class RakutenWordCloud:
    pass