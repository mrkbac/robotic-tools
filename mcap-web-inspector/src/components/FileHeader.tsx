import { Title } from "@mantine/core";

interface FileHeaderProps {
  fileName: string;
}

export function FileHeader({ fileName }: FileHeaderProps) {
  return <Title order={2}>{fileName}</Title>;
}
