
from typing import Literal

class site_tools:

    class wikipedia:

        @staticmethod
        def description(soup) -> str:

            if not (desc := soup.find('h2', id='Description', string='Description')):
                return ''

            iter_ = desc.parent

            paragraphs: list[str] = []

            while (next_p := iter_.find_next_sibling()) and next_p.name == 'p':

                paragraphs.append(next_p.text)

                iter_ = next_p
            
            return "\n".join(paragraphs)

        @staticmethod
        def common_names(soup) -> list[str]:

            # Find first paragraph
            body = soup.find(class_='mw-content-ltr')

            return [b.text for b in body.find_all(lambda tag: tag.name == 'p' and not tag.has_attr('class'))[0].find_all('b')][1:]

        @staticmethod
        def valid_url(soup) -> bool:
            
            return not ((mbox := soup.find(class_='mbox-text')) and 'Wikipedia does not have an article with this exact name.' in mbox.text)

        @staticmethod
        def search_format(genus, species) -> str:

            return f'https://en.wikipedia.org/w/index.php?title={genus.replace(" ", "_")}_{species.replace(" ", "_")}'

    class ncsu:

        @staticmethod
        def description(soup) -> str: 
            
            return dt.find_next_sibling("dd").text \
                if (dt := soup.find("dt", string="Description")) else ''

        @staticmethod
        def common_names(soup) -> list[str]:

            return [name.text for name in soup.find(id='common_names').find_all('li')] \
                if soup.find(id='common_names') else []

        @staticmethod
        def valid_url(soup) -> bool:

            return '404 Not Found' not in soup.title.text

        @staticmethod
        def search_format(genus, species) -> str:

            return f'https://plants.ces.ncsu.edu/plants/{genus.replace(" ", "-")}-{species.replace(" ", "-")}/'

    class britannica:

        @staticmethod
        def valid_url(soup) -> bool:

            return not bool(soup.find(class_='md-error-search-box'))

        @staticmethod
        def search_format(common_names) -> list[str]:

            print(urls := [f'https://britannica.com/plant/{cn.replace(" ", "-")}' for cn in common_names])

            return urls

        @staticmethod
        def common_names(soup) -> list[str]:

            return []

        @staticmethod
        def description(soup) -> str: 
            
            return '/n'.join([p.text for p in phd.parent.find_all('p')]) if (phd := soup.find('h1', string='Physical description')) else ''

    class epicgardening:

        @staticmethod
        def search_format(common_names) -> list[str]:

            return [f'https://epicgardening.com/{cn.replace(" ", "-")}/' for cn in common_names]

        @staticmethod
        def common_names(soup) -> list[str]:

            return []
