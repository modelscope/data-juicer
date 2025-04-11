from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

bs4 = LazyLoader('bs4', 'bs4')

OP_NAME = 'extract_tables_from_html_mapper'


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractTablesFromHtmlMapper(Mapper):
    """Mapper to extract tables from HTML content."""

    def __init__(self,
                 tables_field_name: str = MetaKeys.html_tables,
                 retain_html_tags: bool = False,
                 include_header: bool = True,
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param tables_field_name: Field name to store the extracted tables.
        :param retain_html_tags: If True, retains HTML tags in the tables;
                                 otherwise, removes them.
        :param include_header: If True, includes the table header;
                                otherwise, excludes it.
                This parameter is effective
                            only when `retain_html_tags` is False
                and applies solely to the extracted table content.
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.tables_field_name = tables_field_name
        self.retain_html_tags = retain_html_tags
        self.include_header = include_header

    def process_single(self, sample):
        # check if it's generated already
        if self.tables_field_name in sample[Fields.meta]:
            return sample

        # parse the HTML content using BeautifulSoup
        soup = bs4.BeautifulSoup(sample[self.text_key], 'html.parser')
        tables = soup.find_all('table')

        # if no tables are found, return an empty list
        if not tables:
            sample[Fields.meta][self.tables_field_name] = []
            return sample

        # if retaining HTML tags, store the raw table elements
        if self.retain_html_tags:
            sample[Fields.meta][self.tables_field_name] = [
                str(table) for table in tables
            ]
            return sample

        # extract table data without HTML tags
        extracted_tables = []
        for table in tables:
            extracted_rows = []
            for row in table.find_all('tr'):
                is_header_row = row.find('th', recursive=False) is not None

                # skip rows based on the include_header flag
                if not self.include_header and is_header_row:
                    continue

                # extract text content from cells
                row_data = [
                    cell.get_text(strip=True)
                    for cell in row.find_all(['td', 'th'], recursive=False)
                ]
                if row_data:
                    extracted_rows.append(row_data)

            if extracted_rows:
                extracted_tables.append(extracted_rows)

        sample[Fields.meta][self.tables_field_name] = extracted_tables
        return sample
